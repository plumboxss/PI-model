import os
from collections import defaultdict
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def get_model_id_and_train_test_split(dataset, train_split_path, test_split_path, num_models_to_use=None):
    """
    Get model ids and train/test splits.
    """
    return None, None, None

def get_datasets(dataset_path, observation_dim, action_dim, batch_size=4, set_size=-1, encoder_type='attention'):
    """
    Get train and test datasets.
    """
    # Load dataset using pickle
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # Wrap in PyTorch Dataset
    # For now, use all data for both train/test for simplicity, or implement simple split
    # Let's split 80/20
    total_queries = len(dataset["labels"])
    train_size = int(0.8 * total_queries)
    
    train_indices = list(range(0, train_size))
    test_indices = list(range(train_size, total_queries))
    
    train_dataset = PreferenceDataset(dataset, indices=train_indices)
    test_dataset = PreferenceDataset(dataset, indices=test_indices)

    # num_workers=0 is safer for Windows
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Train queries: {len(train_dataset)}")
    print(f"Test queries: {len(test_dataset)}")
    
    # Return auxiliary info needed by train script
    len_set = 0 
    len_query = 0
    
    # Calculate encoder input dim: obs + act + [label?]
    # For attention encoder, it takes (s1, s2, label) -> pair feature
    # s_dim = obs + act
    # Pair dim = s_dim * 2 + 1 (label) or something
    # VAEModel.__init__ uses encoder_input_dim to init encoder.
    # If attention encoder:
    # It expects input to pair_encoder.
    # Pair encoder input dim = (obs + act) * 2 + 1 (label)
    # wait, VAEModel code:
    # self.encoder = SelfAttentionEncoder(input_dim=encoder_input_dim, ...)
    # Inside SelfAttentionEncoder:
    # self.pair_encoder = PairEncoder(state_dim, action_dim)
    # Ah, SelfAttentionEncoder doesn't use 'input_dim' directly for pair encoder structure,
    # BUT it might use it for transformer input projection?
    # Let's check VAEModel again.
    # It passes encoder_input_dim to SelfAttentionEncoder.
    # SelfAttentionEncoder uses it?
    # Actually, let's look at src/models/vae.py content if possible.
    # But assuming standard flow:
    # encoder_input_dim usually refers to the dimension of the token fed into the transformer.
    # If PairEncoder outputs 'hidden_dim', then transformer input is 'hidden_dim'.
    
    # Let's set encoder_input_dim correctly based on VAE implementation.
    # If we don't have access, safe guess is (obs+act)*2 + 1.
    encoder_input_dim = (observation_dim + action_dim) * 2 + 1

    return train_loader, test_loader, train_dataset, test_dataset, len_set, len_query, encoder_input_dim

class PreferenceDataset(Dataset):
    def __init__(self, pref_dataset, indices=None):
        self.pref_dataset = pref_dataset
        self.indices = indices if indices is not None else list(range(len(pref_dataset['labels'])))
        
        # final_dataset structure from build_preference_dataset.py:
        # 'observations': (N, 1, T, D) -> Wait, axis 1 was expanded?
        # In build_pref:
        # final_dataset[k] = np.expand_dims(np.array(final_dataset[k]), axis=1)
        # So it's (N, 1, T, D). 
        # But we have pairs.
        # Ah, 'observations' is pref_traj, 'observations_2' is non_pref_traj.
        # So we have 4 keys: observations, actions, observations_2, actions_2.
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Get data
        # Dimensions: (1, T, D)
        obs1 = self.pref_dataset['observations'][real_idx][0] 
        act1 = self.pref_dataset['actions'][real_idx][0]
        obs2 = self.pref_dataset['observations_2'][real_idx][0]
        act2 = self.pref_dataset['actions_2'][real_idx][0]
        label = self.pref_dataset['labels'][real_idx][0] # Scalar (1.0)

        # Concatenate obs and act
        # s1: (T, O+A)
        s1 = np.concatenate([obs1, act1], axis=-1)
        s2 = np.concatenate([obs2, act2], axis=-1)
        
        return {
            "s1": s1,
            "s2": s2,
            "labels": label, # Expected shape (1,) or scalar? VAE usually expects (1,)
            "obs1": obs1,
            "act1": act1,
            "obs2": obs2,
            "act2": act2
        }
