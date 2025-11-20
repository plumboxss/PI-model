import math
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def get_datasets(
    dataset_path, observation_dim, action_dim, batch_size, set_size, encoder_type='mlp'
):
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # For VPL, observations are already (obs, action) concatenated
    # In our case, we need to handle obs and actions separately
    obs_dim = dataset["observations"].shape[-1]
    act_dim = dataset["actions"].shape[-1]
    input_dim = obs_dim + act_dim

    # The original VPL code reshapes the dataset based on set_size (context length).
    # Let's see if our dataset is already in the right format.
    # Our data format: (Num_Pairs, 1, Traj_len, Dim)
    # VPL expected: (Num_Queries, Pairs_per_query, Traj_len, Dim)
    # We can treat Num_Pairs as Num_Queries and Pairs_per_query as 1 for MLP.
    # For Attention, we need to group them.

    model_ids = dataset.get('model_id')

    # action_dim을 PreferenceDataset에 전달하기 위해 데이터셋에서 직접 추출합니다.
    act_dim = dataset["actions"].shape[-1]

    train_dataset = PreferenceDataset(dataset, act_dim, model_ids, train=True)
    eval_dataset = PreferenceDataset(dataset, act_dim, model_ids, train=False)
    
    if encoder_type == 'attention':
        # For attention, we group by model_id to form batches.
        # set_size here will be the context length, which is determined by the sampler.
        # We need a collate_fn to stack the pairs into a context.
        
        # The sampler will provide indices for a batch. The default collate_fn
        # will stack them, resulting in (Batch, 1, Traj, Dim).
        # We need to reshape this for the attention encoder.
        # Let's adjust the loader and train script instead.
        # The sampler is the most important part.

        train_sampler = GroupBatchSampler(train_dataset.model_ids, batch_size)
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_sampler, num_workers=4
        )
        
        # For evaluation, we can still shuffle normally
        eval_loader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        
        # For attention encoder, pair_dim = Traj_len * (Obs+Act) + 1
        traj_len = dataset['observations'].shape[2]
        pair_dim = traj_len * input_dim * 2 + 1 
        # But our SelfAttentionEncoder expects one pair at a time... let's rethink.
        # The input to SelfAttentionEncoder is (B, S, pair_dim)
        # where S is seq_len (context size) and pair_dim is flattened (s1,s2,y)
        # The data loader should yield batches of shape (B, S, D) where B is num_contexts
        # and S is context_len. This is complex.

        # Let's simplify:
        # 1. The Sampler ensures a batch has ONE model_id.
        # 2. The default collate_fn creates a batch: (Batch_size, 1, T, D) for obs/act
        # 3. In the train script, we treat this Batch_size as the context length.
        #    We will need to unsqueeze(0) to create a batch of 1 context.
        # This seems like the most straightforward change.
        # get_datasets will just return the loader with the GroupBatchSampler.
        
        # We also need to define the input dimension for the encoder.
        # The SelfAttentionEncoder embeds each pair. The pair dim is flattened s1, s2, label
        # s1, s2는 이제 obs, act가 합쳐진 형태이므로 input_dim을 사용합니다.
        s1_dim = dataset['observations'].shape[2] * (dataset['observations'].shape[3] + act_dim)
        pair_dim_for_attn = s1_dim * 2 + 1
        
        # The original `set_size` is not used in this path.
        # len_set = batch_size, len_query = 1
        return train_loader, eval_loader, train_dataset, eval_dataset, batch_size, 1, pair_dim_for_attn

    else: # Original MLP path
        num_queries_train = train_dataset.num_queries
        num_queries_test = eval_dataset.num_queries
        len_set = dataset["observations"].shape[1]
        len_query = 1

        # MLP 경로는 더 이상 사용되지 않으므로, obs_dim 계산을 Attention 경로와 유사하게 맞춥니다.
        # 이 부분은 train.py에서 MLP 로직이 제거되면 함께 정리될 수 있습니다.
        dataset_obs_dim = (dataset["observations"].shape[-1] + act_dim) * dataset["observations"].shape[2]

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        return (
            train_loader,
            eval_loader,
            train_dataset,
            eval_dataset,
            len_set,
            len_query,
            dataset_obs_dim,
        )


class PreferenceDataset(Dataset):
    def __init__(self, pref_dataset, action_dim, model_ids, train=True):
        self.pref_dataset = pref_dataset
        self.action_dim = action_dim
        self.model_ids = model_ids
        self.train = train
        self.num_queries = self.pref_dataset["observations"].shape[0]
        self.num_total_pairs_per_query = self.pref_dataset["observations"].shape[1]

    def __len__(self):
        return self.num_queries

    def __getitem__(self, query_idx):
        # In VPL, annotation size is the number of pairs per query.
        # For our case, it's the context length.
        num_pairs = self.num_total_pairs_per_query
        
        obs1 = self.pref_dataset["observations"][query_idx, :num_pairs]
        obs2 = self.pref_dataset["observations_2"][query_idx, :num_pairs]
        actions1 = self.pref_dataset["actions"][query_idx, :num_pairs]
        actions2 = self.pref_dataset["actions_2"][query_idx, :num_pairs]
        
        labels = self.pref_dataset["labels"][query_idx, :num_pairs]
        model_id = self.model_ids[query_idx] if self.model_ids is not None else -1

        # encoder 입력을 위해 obs와 action을 합치되,
        # decoder를 위해 원본 obs와 action도 별도로 반환합니다.
        s1 = np.concatenate([obs1, actions1], axis=-1)
        s2 = np.concatenate([obs2, actions2], axis=-1)

        return {
            "s1": s1,
            "s2": s2,
            "obs1": obs1,
            "act1": actions1,
            "obs2": obs2,
            "act2": actions2,
            "labels": labels,
            "model_id": model_id
        }


class GroupBatchSampler(Sampler):
    """
    Custom sampler to ensure each batch contains data from only one user group (model_id).
    """
    def __init__(self, model_ids, batch_size):
        self.model_ids = model_ids
        self.batch_size = batch_size
        
        # Group indices by model_id
        self.indices_by_group = {}
        for idx, model_id in enumerate(model_ids):
            if model_id not in self.indices_by_group:
                self.indices_by_group[model_id] = []
            self.indices_by_group[model_id].append(idx)
        
        self.groups = list(self.indices_by_group.keys())
        self.num_batches = len(model_ids) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            # 1. Randomly select a user group
            group_id = np.random.choice(self.groups)
            
            # 2. Sample a batch of indices from that group
            group_indices = self.indices_by_group[group_id]
            batch_indices = np.random.choice(group_indices, self.batch_size, replace=False)
            
            yield batch_indices

    def __len__(self):
        return self.num_batches


class Annealer:
    def __init__(self, total_steps, shape="cosine", baseline=0.0, cyclical=False):
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if self.shape == "none":
            self.shape = "none"
            self.baseline = 0.0

    def __call__(self, kld):
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == "linear":
            y = self.current_step / self.total_steps
        elif self.shape == "cosine":
            y = (1 - np.cos(self.current_step * np.pi / self.total_steps)) / 2
        elif self.shape == "logistic":
            exponent = (self.total_steps / 2) - self.current_step
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == "none":
            y = 1.0
        else:
            raise ValueError(
                "Invalid shape for annealing function. Must be linear, cosine, or logistic."
            )
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError(
                "Cyclical_setter method requires boolean argument (True/False)"
            )
        else:
            self.cyclical = value
        return


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_posterior(env, reward_model, dataset, mode, num_samples):
    batch, num_samples = dataset.get_mode_data(num_samples)
    return get_latent(batch, env, reward_model, mode, num_samples)


def get_all_posterior(env, reward_model, dataset, num_samples):
    means = []
    for mode in range(env.get_num_modes()):
        means.append(get_posterior(env, reward_model, dataset, mode, num_samples))
    return np.stack(means, axis=0)


def get_biased(env, reward_model, dataset=None):
    means = []
    if dataset:
        batch, _ = dataset.get_mode_data(1)
    else:
        obs1, obs2 = env.get_biased_data(reward_model.annotation_size)
        batch = dict(
            observations=obs1[None, :, None],
            observations_2=obs2[None, :, None],
        )
    # import pdb; pdb.set_trace()
    for mode in range(env.get_num_modes()):
        means.append(get_latent(batch, env, reward_model, mode, 1))
    return np.stack(means, axis=0)
