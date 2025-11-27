import argparse
import numpy as np
from tqdm import tqdm

class SimulationRecorder:
    def __init__(self, env, controller, downsample=1):
        self.env = env
        self.controller = controller
        self.downsample = downsample
        self.results = {
            'time': [], 'state_all': [], 'state_ddot_all': [],
            'state_dddot_all': [], 'action_all': [], 'reward_all': [],
        }

    def simulate(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        done = False
        
        # Use params.T directly
        T = getattr(self.env.plant.params, 'T', 10.0)
        time_step = self.env.time_step
        num_steps = int(T / time_step)
        
        with tqdm(total=num_steps, desc="Simulating Episode", leave=False) as pbar:
            while not done:
                action = self.controller.control(obs)
                obs, reward, done, info = self.env.step(action)
                self.results['time'].append(self.env.current_time)
                self.results['state_all'].append(obs)
                self.results['action_all'].append(action)
                self.results['reward_all'].append(reward)
                self.results['state_ddot_all'].append(info.get('state_ddot', np.zeros_like(obs)))
                self.results['state_dddot_all'].append(info.get('state_dddot', np.zeros_like(obs)))
                pbar.update(1)
        for key, val in self.results.items():
            self.results[key] = np.array(val)
        if self.downsample > 1:
            self.results['state'] = self.results['state_all'][::self.downsample]
            self.results['state_ddot'] = self.results['state_ddot_all'][::self.downsample]
        else:
            self.results['state'] = self.results['state_all']
            self.results['state_ddot'] = self.results['state_ddot_all']

    def __getitem__(self, key):
        return self.results[key]

def get_trajectory_features(recorder, settling_time_threshold=0.01):
    """Extracts scalar features from a trajectory."""
    # Ensure inputs are numpy arrays for consistent handling
    time = np.array(recorder['time'])
    states = np.array(recorder['state_all'])
    actions = np.array(recorder['action_all'])

    # 1. Jerk
    # Check for empty or too short trajectories to avoid errors
    if len(time) < 2:
        return {"jerk": 0.0, "pitch": 0.0, "settling_time": 0.0}

    jerk = np.mean(np.abs(np.diff(actions, n=2, axis=0) / np.diff(time[:-1])[:, np.newaxis]))

    # 2. Pitch
    pitch = np.mean(np.abs(states[:, 6])) # 6 is the index for theta (pitch)

    # 3. Settling Time
    pitch_velocity = states[:, 1] # 1 is the index for dtheta (pitch velocity)
    final_pitch_velocity = pitch_velocity[-10:]
    settled = np.all(np.abs(final_pitch_velocity) < settling_time_threshold)
    
    settling_time = time[-1] # Default to max time
    if settled:
        settled_indices = np.where(np.abs(pitch_velocity) < settling_time_threshold)[0]
        if len(settled_indices) > 0:
            # Find the last time it was unsettled
            unsettled_indices = np.where(np.abs(pitch_velocity) >= settling_time_threshold)[0]
            if len(unsettled_indices) > 0:
                last_unsettled_time = time[unsettled_indices[-1]]
                settling_time = last_unsettled_time
            else: # Already settled from the beginning
                settling_time = 0.0

    return {
        "jerk": float(jerk),
        "pitch": float(pitch),
        "settling_time": float(settling_time)
    }

def generate_data_parser():
    parser = argparse.ArgumentParser(description="Generate trajectory data using vehicle simulation.")
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of simulation episodes to run.')
    parser.add_argument('--oracle-name', type=str, required=True, help="Name of the oracle to use (e.g., 'A', 'B'). Corresponds to 'oracle_A.yaml'.")
    parser.add_argument('--dataset-name', type=str, default=None, help='Name for the generated dataset file.')
    return parser

def visualize_oracle_data(dataset_path, save_dir):
    print(f"[Placeholder] Visualizing data from {dataset_path} and saving to {save_dir}...")
    pass
