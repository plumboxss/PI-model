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

def get_datasets(dataset_path, observation_dim, action_dim, batch_size=4, set_size=-1, encoder_type='attention', context_size=5, split_seed=42):
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
        split_seed: random seed for train/test group split (for reproducibility)
    """
    # Load dataset using pickle
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # Group by model_id (user group) for context sampling
    # If model_id exists, use it; otherwise raise error or check for alternatives
    if 'model_id' in dataset:
        grouped_indices = defaultdict(list)
        for idx in range(len(dataset['labels'])):
            model_id = dataset['model_id'][idx]
            grouped_indices[model_id].append(idx)
    else:
        # Check for alternative grouping keys
        alternative_keys = ['oracle_id', 'driver_id', 'user_id', 'annotator_id']
        found_key = None
        for key in alternative_keys:
            if key in dataset:
                found_key = key
                break
        
        if found_key:
            print(f"Warning: 'model_id' not found, using '{found_key}' for grouping.")
            grouped_indices = defaultdict(list)
            for idx in range(len(dataset['labels'])):
                group_id = dataset[found_key][idx]
                grouped_indices[group_id].append(idx)
        else:
            # No meaningful grouping key found - this breaks few-shot adaptation
            raise ValueError(
                "No 'model_id' (or alternative: 'oracle_id', 'driver_id', 'user_id', 'annotator_id') found in dataset. "
                "Few-shot preference adaptation requires grouping by user/oracle identity. "
                "Random grouping would make z represent noise rather than user preferences."
            )
    
    # Split train/test with shuffle for proper distribution
    all_groups = list(grouped_indices.keys())
    # Use seed-based shuffle for reproducibility
    rng = np.random.RandomState(split_seed)
    shuffled_groups = all_groups.copy()
    rng.shuffle(shuffled_groups)
    
    train_size = int(0.8 * len(shuffled_groups))
    train_groups = shuffled_groups[:train_size]
    test_groups = shuffled_groups[train_size:]
    
    # Train dataset: use random context sampling for diversity
    train_dataset = ContextQueryDataset(
        dataset, grouped_indices, train_groups, context_size,
        deterministic_context=False,  # Random context for training diversity
        deterministic_seed=split_seed
    )
    
    # Test dataset: use deterministic context sampling for evaluation stability
    test_dataset = ContextQueryDataset(
        dataset, grouped_indices, test_groups, context_size,
        deterministic_context=True,  # Deterministic context for evaluation stability
        deterministic_seed=split_seed
    )
    
    # Verify deterministic context sampling (test only)
    if len(test_dataset) > 0:
        # Get first sample twice and verify context is identical
        # This verifies that deterministic_context=True works correctly
        sample1 = test_dataset[0]
        sample2 = test_dataset[0]
        # Check if context tensors are identical (should be for deterministic sampling)
        context_match = torch.equal(sample1['context_s1'], sample2['context_s1'])
        if context_match:
            print(f"✅ Test dataset: deterministic_context=True verified (evaluation stability enabled)")
            print(f"   Same (group_id, query_idx) produces identical context across calls")
        else:
            print(f"⚠️  Warning: Test dataset context is not deterministic (this should not happen)")
    
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

# Global flag to check shapes only once (first batch)
_first_batch_checked = False

def collate_context_query(batch):
    """
    Collate function for context-query batches.
    
    Args:
        batch: list of dicts with keys: context_s1, context_s2, context_y, query_s1, query_s2, query_y
    Returns:
        batched dict with same keys, stacked along batch dimension
    """
    global _first_batch_checked
    
    # Stack context data: (B, K, T, D_sa)
    context_s1 = torch.stack([item['context_s1'] for item in batch])
    context_s2 = torch.stack([item['context_s2'] for item in batch])
    context_y = torch.stack([item['context_y'] for item in batch])
    
    # Stack query data: (B, T, D_sa)
    query_s1 = torch.stack([item['query_s1'] for item in batch])
    query_s2 = torch.stack([item['query_s2'] for item in batch])
    query_y = torch.stack([item['query_y'] for item in batch])
    
    # Verify shapes on first batch only
    if not _first_batch_checked:
        B, K, T, D_sa = context_s1.shape
        assert context_s2.shape == (B, K, T, D_sa), f"context_s2 shape mismatch: {context_s2.shape} vs {(B, K, T, D_sa)}"
        assert context_y.shape == (B, K, 1), f"context_y shape mismatch: {context_y.shape} vs {(B, K, 1)}"
        assert query_s1.shape == (B, T, D_sa), f"query_s1 shape mismatch: {query_s1.shape} vs {(B, T, D_sa)}"
        assert query_s2.shape == (B, T, D_sa), f"query_s2 shape mismatch: {query_s2.shape} vs {(B, T, D_sa)}"
        assert query_y.shape == (B, 1), f"query_y shape mismatch: {query_y.shape} vs {(B, 1)}"
        print(f"✅ First batch shape verification passed: context=(B={B}, K={K}, T={T}, D={D_sa}), query=(B={B}, T={T}, D={D_sa})")
        _first_batch_checked = True
    
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
    - Context: K comparisons from the same user group (sampled dynamically or deterministically)
    - Query: 1 comparison for loss computation
    
    Note: 
    - If deterministic_context=False: Context is sampled randomly in __getitem__ for diversity.
      This prevents overfitting to fixed context sets and better matches few-shot adaptation goals.
    - If deterministic_context=True: Context is sampled deterministically based on (group_id, query_idx).
      This ensures evaluation stability by using the same context for the same sample across epochs.
    """
    def __init__(self, pref_dataset, grouped_indices, group_ids, context_size, 
                 deterministic_context=False, deterministic_seed=0):
        """
        Args:
            pref_dataset: loaded preference dataset dict
            grouped_indices: dict mapping group_id to list of indices
            group_ids: list of group IDs to use (train or test)
            context_size: number of context comparisons (K)
            deterministic_context: if True, use deterministic context sampling (for test/eval stability)
            deterministic_seed: seed for deterministic context sampling (used in hash)
        """
        self.pref_dataset = pref_dataset
        self.grouped_indices = grouped_indices
        self.group_ids = group_ids
        self.context_size = context_size
        self.deterministic_context = deterministic_context
        self.deterministic_seed = deterministic_seed
        
        # Track first call for verification logging
        self._first_call_verified = False
        
        # Create samples: each sample is (group_id, query_idx)
        # Context will be sampled in __getitem__ (randomly or deterministically)
        self.samples = []
        for group_id in group_ids:
            indices = grouped_indices[group_id]
            if len(indices) < context_size + 1:
                # Skip groups with insufficient data
                continue
            
            # Create multiple samples from each group by rotating query
            # Each sample uses a different query_idx, context will be sampled per call
            for query_idx in range(len(indices)):
                self.samples.append((group_id, query_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        group_id, query_idx = self.samples[idx]
        group_indices = self.grouped_indices[group_id]
        
        # Build context candidate pool (excluding query_idx to ensure no overlap)
        context_candidates = [i for i in range(len(group_indices)) if i != query_idx]
        
        if len(context_candidates) < self.context_size:
            # Fallback: if not enough candidates, use all available (with replacement if needed)
            if len(context_candidates) == 0:
                # Edge case: only one sample in group, cannot create context-query split
                raise ValueError(f"Group {group_id} has insufficient data for context-query split")
            # Use all available candidates and repeat if necessary
            context_indices = (context_candidates * ((self.context_size // len(context_candidates)) + 1))[:self.context_size]
        else:
            # Sample context indices (randomly or deterministically)
            if self.deterministic_context:
                # Deterministic sampling: use hash-based seed for reproducibility
                # Same (group_id, query_idx) always produces the same context
                seed = (hash((group_id, query_idx, self.deterministic_seed)) & 0xffffffff)
                rs = np.random.RandomState(seed)
                context_indices = rs.choice(
                    context_candidates,
                    size=self.context_size,
                    replace=False
                ).tolist()
            else:
                # Random sampling: different context each time (for training diversity)
                context_indices = np.random.choice(
                    context_candidates,
                    size=self.context_size,
                    replace=False
                ).tolist()
        
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
