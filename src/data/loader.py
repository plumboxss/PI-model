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

def get_datasets(dataset_path, observation_dim, action_dim, batch_size=4, set_size=-1, encoder_type='attention', context_size=5):
    """
    Get train and test datasets with context-query structure.
    
    Args:
        dataset_path: path to preference dataset
        observation_dim: dimension of observations
        action_dim: dimension of actions
        batch_size: batch size for DataLoader
        set_size: unused (for compatibility)
        encoder_type: unused (for compatibility)
        context_size: number of context comparisons (K)
    """
    # Load dataset using pickle
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # Group by model_id (user group) for context sampling
    # If model_id exists, use it; otherwise group randomly
    if 'model_id' in dataset:
        grouped_indices = defaultdict(list)
        for idx in range(len(dataset['labels'])):
            model_id = dataset['model_id'][idx]
            grouped_indices[model_id].append(idx)
    else:
        # If no model_id, create random groups
        num_groups = max(1, len(dataset['labels']) // context_size)
        grouped_indices = defaultdict(list)
        for idx in range(len(dataset['labels'])):
            group_id = idx % num_groups
            grouped_indices[group_id].append(idx)
    
    # Split train/test
    all_groups = list(grouped_indices.keys())
    train_size = int(0.8 * len(all_groups))
    train_groups = all_groups[:train_size]
    test_groups = all_groups[train_size:]
    
    train_dataset = ContextQueryDataset(dataset, grouped_indices, train_groups, context_size)
    test_dataset = ContextQueryDataset(dataset, grouped_indices, test_groups, context_size)
    
    # num_workers=0 is safer for Windows
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_context_query,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_context_query,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Context size: {context_size}")
    
    # Return auxiliary info (for compatibility)
    len_set = context_size
    len_query = 1
    encoder_input_dim = observation_dim + action_dim  # Not used in new architecture
    
    return train_loader, test_loader, train_dataset, test_dataset, len_set, len_query, encoder_input_dim

def collate_context_query(batch):
    """
    Collate function for context-query batches.
    
    Args:
        batch: list of dicts with keys: context_s1, context_s2, context_y, query_s1, query_s2, query_y
    Returns:
        batched dict with same keys, stacked along batch dimension
    """
    # Stack context data: (B, K, T, D_sa)
    context_s1 = torch.stack([item['context_s1'] for item in batch])
    context_s2 = torch.stack([item['context_s2'] for item in batch])
    context_y = torch.stack([item['context_y'] for item in batch])
    
    # Stack query data: (B, T, D_sa)
    query_s1 = torch.stack([item['query_s1'] for item in batch])
    query_s2 = torch.stack([item['query_s2'] for item in batch])
    query_y = torch.stack([item['query_y'] for item in batch])
    
    return {
        'context_s1': context_s1,
        'context_s2': context_s2,
        'context_y': context_y,
        'query_s1': query_s1,
        'query_s2': query_s2,
        'query_y': query_y,
    }

class ContextQueryDataset(Dataset):
    """
    Dataset that provides context-query structure for few-shot learning.
    
    Each sample consists of:
    - Context: K comparisons from the same user group
    - Query: 1 comparison for loss computation
    """
    def __init__(self, pref_dataset, grouped_indices, group_ids, context_size):
        """
        Args:
            pref_dataset: loaded preference dataset dict
            grouped_indices: dict mapping group_id to list of indices
            group_ids: list of group IDs to use (train or test)
            context_size: number of context comparisons (K)
        """
        self.pref_dataset = pref_dataset
        self.grouped_indices = grouped_indices
        self.group_ids = group_ids
        self.context_size = context_size
        
        # Create samples: each sample is (group_id, context_indices, query_idx)
        # 재현성을 위해 seed 설정
        np.random.seed(42)
        self.samples = []
        for group_id in group_ids:
            indices = grouped_indices[group_id]
            if len(indices) < context_size + 1:
                # Skip groups with insufficient data
                continue
            
            # Create multiple samples from each group by rotating query
            for query_idx in range(len(indices)):
                # Context: randomly sample K indices (excluding query)
                context_candidates = [i for i in range(len(indices)) if i != query_idx]
                if len(context_candidates) < context_size:
                    continue
                context_indices = np.random.choice(
                    context_candidates,
                    size=context_size,
                    replace=False
                ).tolist()
                
                self.samples.append((group_id, context_indices, query_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        group_id, context_indices, query_idx = self.samples[idx]
        group_indices = self.grouped_indices[group_id]
        
        # Get context data
        context_s1_list = []
        context_s2_list = []
        context_y_list = []
        
        for ctx_idx in context_indices:
            real_idx = group_indices[ctx_idx]
            # Get data: (1, T, D) -> (T, D)
            obs1 = self.pref_dataset['observations'][real_idx][0]
            act1 = self.pref_dataset['actions'][real_idx][0]
            obs2 = self.pref_dataset['observations_2'][real_idx][0]
            act2 = self.pref_dataset['actions_2'][real_idx][0]
            label = self.pref_dataset['labels'][real_idx][0]
            
            # Concatenate obs and act
            s1 = np.concatenate([obs1, act1], axis=-1)  # (T, obs+act)
            s2 = np.concatenate([obs2, act2], axis=-1)  # (T, obs+act)
            
            context_s1_list.append(s1)
            context_s2_list.append(s2)
            context_y_list.append(label)
        
        # Get query data
        query_real_idx = group_indices[query_idx]
        query_obs1 = self.pref_dataset['observations'][query_real_idx][0]
        query_act1 = self.pref_dataset['actions'][query_real_idx][0]
        query_obs2 = self.pref_dataset['observations_2'][query_real_idx][0]
        query_act2 = self.pref_dataset['actions_2'][query_real_idx][0]
        query_label = self.pref_dataset['labels'][query_real_idx][0]
        
        query_s1 = np.concatenate([query_obs1, query_act1], axis=-1)  # (T, obs+act)
        query_s2 = np.concatenate([query_obs2, query_act2], axis=-1)  # (T, obs+act)
        
        # Convert to tensors
        context_s1 = torch.from_numpy(np.array(context_s1_list)).float()  # (K, T, obs+act)
        context_s2 = torch.from_numpy(np.array(context_s2_list)).float()  # (K, T, obs+act)
        context_y = torch.from_numpy(np.array(context_y_list)).float().unsqueeze(-1)  # (K, 1)
        
        query_s1 = torch.from_numpy(query_s1).float()  # (T, obs+act)
        query_s2 = torch.from_numpy(query_s2).float()  # (T, obs+act)
        query_y = torch.tensor([query_label]).float()  # (1,)
        
        return {
            'context_s1': context_s1,
            'context_s2': context_s2,
            'context_y': context_y,
            'query_s1': query_s1,
            'query_s2': query_s2,
            'query_y': query_y,
        }

# Legacy dataset for backward compatibility
class PreferenceDataset(Dataset):
    def __init__(self, pref_dataset, indices=None):
        self.pref_dataset = pref_dataset
        self.indices = indices if indices is not None else list(range(len(pref_dataset['labels'])))
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Get data
        obs1 = self.pref_dataset['observations'][real_idx][0] 
        act1 = self.pref_dataset['actions'][real_idx][0]
        obs2 = self.pref_dataset['observations_2'][real_idx][0]
        act2 = self.pref_dataset['actions_2'][real_idx][0]
        label = self.pref_dataset['labels'][real_idx][0]
        
        # Concatenate obs and act
        s1 = np.concatenate([obs1, act1], axis=-1)
        s2 = np.concatenate([obs2, act2], axis=-1)
        
        return {
            "s1": s1,
            "s2": s2,
            "labels": label,
            "obs1": obs1,
            "act1": act1,
            "obs2": obs2,
            "act2": act2
        }
