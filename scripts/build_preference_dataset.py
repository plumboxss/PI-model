import os
import pickle
import argparse
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

def main(args):
    # 1. Load data and extract features
    print(f"Loading raw trajectory data from {args.input_path}...")
    with open(args.input_path, 'rb') as f:
        raw_data = pickle.load(f)

    features_list = []
    trajectories = []
    for i in sorted(raw_data.keys()):
        result = raw_data[i]
        if 'features' in result and result['features'] is not None:
            feature_vector = np.array([
                result['features']['avg_jerk'],
                result['features']['max_jerk'],
                result['features']['settle_time'],
                result['features']['max_pitch']
            ])
            features_list.append(feature_vector)
            trajectories.append({
                'observations': result['state'],
                'actions': result['action'],
            })
    
    features_matrix = np.array(features_list)
    print(f"Extracted {len(trajectories)} trajectories with features.")

    # 2. Unsupervised Clustering
    print(f"Performing K-Means clustering with K={args.num_clusters}...")
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_matrix)
    print(f"Clustering complete. Found {len(np.unique(cluster_labels))} clusters.")

    # Create index lists for each cluster
    trajectories_by_cluster = [[] for _ in range(args.num_clusters)]
    for idx, label in enumerate(cluster_labels):
        trajectories_by_cluster[label].append(idx)

    # 3. Generate Preference Dataset
    print(f"Generating {args.num_pairs} preference pairs for {args.num_clusters} user groups...")
    
    final_dataset = {
        'observations': [], 'actions': [],
        'observations_2': [], 'actions_2': [],
        'labels': [], 'model_id': []
    }
    
    num_trajectories = len(trajectories)
    min_len = min(len(t['observations']) for t in trajectories) if num_trajectories > 0 else 0
    if min_len == 0:
        print("No valid trajectories found. Exiting.")
        return
    print(f"All trajectories will be truncated to the minimum length of {min_len}.")

    for _ in tqdm(range(args.num_pairs)):
        # a. Select a user group (a cluster)
        group_id = np.random.randint(args.num_clusters)
        
        # Ensure the selected cluster is not empty
        if not trajectories_by_cluster[group_id]:
            continue

        # b. Sample a trajectory from inside the group
        idx_in = np.random.choice(trajectories_by_cluster[group_id])
        
        # c. Sample a trajectory from outside the group
        # Create a list of all indices not in the current group
        outside_indices = [i for i in range(num_trajectories) if cluster_labels[i] != group_id]
        if not outside_indices:
            continue # Should not happen if there's more than one cluster with members
        idx_out = np.random.choice(outside_indices)

        traj_in = trajectories[idx_in]
        traj_out = trajectories[idx_out]

        # d. The "in-group" trajectory is always preferred (label=1.0)
        label = 1.0
        
        # e. Append to dataset, with truncation
        # The preferred trajectory (in-group) is trajectory 1
        final_dataset['observations'].append(traj_in['observations'][:min_len])
        final_dataset['actions'].append(traj_in['actions'][:min_len])
        
        # The non-preferred trajectory (out-group) is trajectory 2
        final_dataset['observations_2'].append(traj_out['observations'][:min_len])
        final_dataset['actions_2'].append(traj_out['actions'][:min_len])

        final_dataset['labels'].append(label)
        final_dataset['model_id'].append(group_id)

    # 4. Convert to VPL compatible format (numpy arrays) and save
    print("Converting to final format and saving...")
    for k in ['observations', 'actions', 'observations_2', 'actions_2']:
        final_dataset[k] = np.expand_dims(np.array(final_dataset[k]), axis=1)
    
    final_dataset['labels'] = np.expand_dims(np.array(final_dataset['labels']), axis=1)
    final_dataset['model_id'] = np.array(final_dataset['model_id'])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(final_dataset, f)
    
    print(f"Preference dataset saved to {args.output_path}")
    print("Dataset shapes:")
    for k, v in final_dataset.items():
        print(f"  {k}: {v.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build preference dataset from trajectory features.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the raw trajectory data pkl file from generate_data.py.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the final preference dataset pkl file.')
    parser.add_argument('--num_clusters', type=int, default=16, help='Number of user groups to cluster trajectories into (must be > 10).')
    parser.add_argument('--num_pairs', type=int, default=20000, help='Number of preference pairs to generate.')
    args = parser.parse_args()
    if args.num_clusters <= 10:
        raise ValueError("num_clusters must be greater than 10.")
    main(args)
