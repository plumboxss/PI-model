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
    pitch_mean = np.mean(np.abs(states[:, 6])) # 6 is the index for theta (pitch)

    # 3. Settling Time
    # Settling time: when the RMS of pitch angle over a window falls below 5% of its initial RMS
    # This measures decay of oscillation energy and is more sensitive to damping differences.
    pitch_angle = states[:, 6]  # 6 is the index for theta (pitch angle)

    # Window size: ~4% of trajectory length (at least 5 timesteps) for smoother RMS
    window_size = max(5, len(pitch_angle) // 25)

    # Compute RMS over sliding window
    def rms_window(arr, start, end):
        segment = arr[start:end]
        return np.sqrt(np.mean(segment * segment))

    # Initial RMS from the first window (avoid near-zero divide)
    initial_rms = rms_window(pitch_angle, 0, window_size)
    if initial_rms < 1e-8:
        settling_time = 0.0
    else:
        threshold = 0.12 * initial_rms  # 12% of initial RMS (more lenient to spread settling times)
        settling_idx = len(time) - 1  # default to end

        for i in range(len(pitch_angle) - window_size + 1):
            cur_rms = rms_window(pitch_angle, i, i + window_size)
            if cur_rms < threshold:
                # ensure it stays low for the remainder (or next window)
                # check next window as simple stability check
                if i + window_size < len(pitch_angle):
                    next_rms = rms_window(pitch_angle, i + 1, min(len(pitch_angle), i + 1 + window_size))
                    if next_rms < threshold:
                        settling_idx = i
                        break
                else:
                    settling_idx = i
                    break

        settling_time = time[settling_idx]

    return {
        "jerk": float(jerk),
        "pitch": float(pitch_mean),
        "settling_time": float(settling_time)
    }

def generate_data_parser():
    parser = argparse.ArgumentParser(description="Generate trajectory data using vehicle simulation.")
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of simulation episodes to run.')
    parser.add_argument('--dataset-id', type=str, default='A', help="Dataset ID for organizing output files (e.g., 'A', 'B').")
    parser.add_argument('--dataset-name', type=str, default=None, help='Name for the generated dataset file.')
    return parser

