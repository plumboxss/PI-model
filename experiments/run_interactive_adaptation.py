import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm
import os
from functools import partial

# Assuming VPL project structure
# Ensure pref_learn is in the python path
import sys
sys.path.append(os.getcwd())
from pref_learn.models.vae import VAEModel

def calculate_state_rewards(states_b, z, comparison_set_C, vae_model):
    """
    Calculates the normalized reward for a batch of states.
    r(s,z) = 1/|C| * sum_{s' in C} P(s > s' | z)
    """
    with torch.no_grad():
        T = states_b.shape[0] # Number of states in the batch
        N = comparison_set_C.shape[0] # Size of the comparison set
        
        # Prepare tensors for batch computation
        # Repeat states to compare against each element in C: (T, N, D)
        states_b_rpt = states_b.unsqueeze(1).repeat(1, N, 1)
        # Repeat C for each state in the batch: (T, N, D)
        C_rpt = comparison_set_C.unsqueeze(0).repeat(T, 1, 1)
        # Repeat z for all comparisons: (T * N, Z)
        z_rpt = z.repeat(T * N, 1)

        # Flatten for model input
        states_flat = states_b_rpt.view(T * N, -1)
        C_flat = C_rpt.view(T * N, -1)

        # Get raw reward scores from the VAE decoder
        r_states = vae_model.decode(states_flat, z_rpt)
        r_C = vae_model.decode(C_flat, z_rpt)

        # Calculate preference probabilities P(s > s' | z)
        probs_flat = torch.sigmoid(r_states - r_C)
        
        # Reshape to (T, N) to average over N
        probs = probs_flat.view(T, N)
        
        # The normalized reward for each state is the mean of these probabilities
        state_rewards = probs.mean(dim=1) # Shape: (T,)
    
    return state_rewards

def create_trajectory_scorer(z_current, vae_model, comparison_set_C):
    """
    Creates a function that scores a trajectory based on the average normalized reward of its states.
    R(sigma, z) = 1/|sigma| * sum_{s in sigma} r(s,z)
    """
    device = next(vae_model.parameters()).device

    def trajectory_scorer(trajectory):
        # Extract observations and move to the correct device
        obs_np = trajectory['observations']
        obs_tensor = torch.from_numpy(obs_np).float().to(device)

        # Calculate normalized rewards for all states in the trajectory
        state_rewards = calculate_state_rewards(obs_tensor, z_current, comparison_set_C, vae_model)
        
        # The trajectory score is the mean of its state rewards
        return state_rewards.mean().item()

    return trajectory_scorer


def find_implicit_pair(input_traj, input_traj_features, trajectory_scorer, 
                       trajectories, features_matrix, epsilon=0.1, search_sample_size=500):
    """
    Finds the best implicit trajectory to form a preference pair.
    """
    # For efficiency, we search over a random subset of the dataset
    search_indices = np.random.choice(len(trajectories), search_sample_size, replace=False)
    
    best_candidate_idx = -1
    min_uncertainty_score = float('inf')

    # Calculate the score for the input trajectory once
    score_input = trajectory_scorer(input_traj)

    for idx in tqdm(search_indices, desc="Finding implicit pair", leave=False):
        candidate_traj = trajectories[idx]
        candidate_features = features_matrix[idx]
        
        # 1. Check diversity condition (using avg_jerk: feature index 0)
        diversity_score = np.abs(input_traj_features[0] - candidate_features[0])
        if diversity_score < epsilon:
            continue
            
        # 2. Calculate uncertainty using the trajectory scorer
        score_candidate = trajectory_scorer(candidate_traj)
        
        # Calculate preference probability P(input > candidate) using trajectory scores
        prob_input_gt_cand = 1 / (1 + np.exp(-(score_input - score_candidate)))

        uncertainty_score = abs(prob_input_gt_cand - 0.5)
        
        if uncertainty_score < min_uncertainty_score:
            min_uncertainty_score = uncertainty_score
            best_candidate_idx = idx

    if best_candidate_idx == -1:
        print("Warning: No diverse enough trajectory found. Returning a random one.")
        random_idx = np.random.choice(len(trajectories))
        return trajectories[random_idx]
        
    return trajectories[best_candidate_idx]


class AdaptationLoop:
    def __init__(self, args):
        self.args = args
        
        # 1. Load pretrained VAE model
        print("Loading pretrained VAE model...")
        self.vae_model = torch.load(args.vae_model_path)
        self.vae_model.eval()
        self.device = next(self.vae_model.parameters()).device
        print("Model loaded successfully.")

        # 2. Load full trajectory dataset and create the fixed comparison set C
        print("Loading trajectory dataset...")
        with open(args.trajectory_dataset_path, 'rb') as f:
            raw_data = pickle.load(f)

        self.trajectories = []
        self.features_list = []
        all_states_list = []
        for i in sorted(raw_data.keys()):
            res = raw_data[i]
            if res.get('features'):
                self.trajectories.append({'observations': res['state'], 'actions': res['action']})
                self.features_list.append(np.array(list(res['features'].values())))
                all_states_list.append(res['state'])

        self.features_matrix = np.array(self.features_list)
        all_states = np.concatenate(all_states_list, axis=0)
        print(f"Loaded {len(self.trajectories)} trajectories and {len(all_states)} total states.")

        # Create the fixed comparison set C by sampling states
        print(f"Creating fixed comparison set C with size {args.comparison_set_size}...")
        comparison_indices = np.random.choice(len(all_states), args.comparison_set_size, replace=False)
        self.comparison_set_C = torch.from_numpy(all_states[comparison_indices]).float().to(self.device)

        # 3. Initialize z and context
        self.z_current = torch.randn(1, self.vae_model.latent_dim, device=self.device)
        self.context = [] # List to store (traj1_obs, traj2_obs, label)
        print(f"Initialized z with shape: {self.z_current.shape}")


    def step(self, input_traj, input_traj_features, input_label):
        """Performs one step of the adaptation loop."""
        print(f"\n--- Step {len(self.context) + 1} ---")
        
        # 1. Create the trajectory scorer function based on the current z and fixed set C
        print("1. Creating trajectory scorer for current z...")
        trajectory_scorer = create_trajectory_scorer(self.z_current, self.vae_model, self.comparison_set_C)

        # 2. Find implicit pair
        print("2. Searching for an implicit pair...")
        implicit_traj = find_implicit_pair(input_traj, input_traj_features, trajectory_scorer, self.trajectories, self.features_matrix, epsilon=self.args.diversity_epsilon)
        print("Implicit pair found.")

        # 3. Update context
        # Assuming input_label=1 means input_traj is preferred
        if input_label == 1:
            self.context.append((input_traj['observations'], implicit_traj['observations'], 1.0))
        else:
            self.context.append((implicit_traj['observations'], input_traj['observations'], 1.0))
        print(f"3. Context updated. Current context size: {len(self.context)}")

        # 4. Re-infer z using the entire context
        print("4. Re-inferring z from the updated context...")
        
        # Prepare the whole context as a single batch for the attention encoder
        obs1_list, obs2_list, labels_list = zip(*self.context)
        
        # Find the minimum trajectory length in the context for padding/truncating
        min_len_obs1 = min(o.shape[0] for o in obs1_list)
        min_len_obs2 = min(o.shape[0] for o in obs2_list)
        min_len = min(min_len_obs1, min_len_obs2)

        # Truncate all trajectories to the minimum length and stack
        obs1_batch = np.array([o[:min_len] for o in obs1_list])
        obs2_batch = np.array([o[:min_len] for o in obs2_list])
        labels_batch = np.array(labels_list)

        # Add a batch dimension (for a single context) and a pair dimension
        obs1_tensor = torch.from_numpy(obs1_batch).float().to(self.device).unsqueeze(1)
        obs2_tensor = torch.from_numpy(obs2_batch).float().to(self.device).unsqueeze(1)
        labels_tensor = torch.from_numpy(labels_batch).float().to(self.device).view(1, -1, 1)

        # Now we need to handle the fact that s1/s2 are obs+act.
        # Let's assume actions are zeros for now as we don't have them for adaptation.
        # This needs to be consistent with how the pretraining dataset was built.
        # Our dataset combines obs+act inside the PreferenceDataset class.
        # So the vae_model.encode expects concatenated obs+act. Let's create dummy actions.
        
        dummy_actions1 = torch.zeros_like(obs1_tensor[..., :1]) # Assuming action dim is 1
        dummy_actions2 = torch.zeros_like(obs2_tensor[..., :1])

        s1_context = torch.cat([obs1_tensor, dummy_actions1], dim=-1)
        s2_context = torch.cat([obs2_tensor, dummy_actions2], dim=-1)

        # The shape should be (B, C, T, D) -> (1, Context_len, T, D)
        # Reshape to (1, num_pairs, traj_len, obs_dim+act_dim)
        context_len = len(self.context)
        s1_context = s1_context.view(1, context_len, min_len, -1)
        s2_context = s2_context.view(1, context_len, min_len, -1)

        with torch.no_grad():
            mean, log_var = self.vae_model.encode(s1_context, s2_context, labels_tensor)
            self.z_current = self.vae_model.reparameterization(mean, torch.exp(0.5 * log_var))
        
        print(f"z re-inferred successfully. New z mean: {mean.mean().item():.4f}")
        return self.z_current

    def run(self):
        """Runs the interactive adaptation session."""
        print("\nStarting interactive adaptation loop.")
        print("In each step, provide a trajectory and your preference (1 for good, 0 for bad).")
        
        # In a real scenario, this loop would be driven by external inputs.
        # Here, we simulate it with a few dummy inputs.
        for i in range(3): # Simulate 3 steps of feedback
            # Dummy input: just pick a random trajectory from the dataset
            dummy_input_idx = np.random.randint(len(self.trajectories))
            dummy_input_traj = self.trajectories[dummy_input_idx]
            dummy_input_features = self.features_matrix[dummy_input_idx]
            dummy_label = np.random.choice([0, 1])

            print(f"\nSimulating user input: Trajectory {dummy_input_idx}, Preference: {dummy_label}")
            self.step(dummy_input_traj, dummy_input_features, dummy_label)
        
        print("\nAdaptation finished.")
        print("Final z (mean of distribution):")
        print(self.z_current)

        # Save the final z
        output_path = self.args.output_z_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(self.z_current, output_path)
        print(f"Final z saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run interactive adaptation to find a user's latent preference z.")
    parser.add_argument('--vae_model_path', type=str, required=True, help='Path to the pretrained VAE model (.pt file).')
    parser.add_argument('--trajectory_dataset_path', type=str, required=True, help='Path to the raw trajectory dataset (.pkl file).')
    parser.add_argument('--output_z_path', type=str, default='data/adapted_z.pt', help='Path to save the final adapted z vector.')
    parser.add_argument('--comparison_set_size', type=int, default=1000, help='Number of states for the fixed comparison set C.')
    parser.add_argument('--diversity_epsilon', type=float, default=0.1, help='Minimum feature difference for a diverse pair.')

    args = parser.parse_args()
    
    # Example usage (requires a trained model and a dataset):
    # python experiments/run_interactive_adaptation.py --vae_model_path logs/my_model/best_model.pt --trajectory_dataset_path artifacts/A/datasets/raw_trajectories.pkl
    
    loop = AdaptationLoop(args)
    loop.run()
