import argparse
import numpy as np
try:
    from tqdm import tqdm
except Exception:
    # Fallback when tqdm isn't installed (progress bar becomes a no-op iterator).
    def tqdm(x=None, **kwargs):
        if x is None:
            class _Dummy:
                def __enter__(self): return self
                def __exit__(self, exc_type, exc, tb): return False
                def update(self, n=1): pass
            return _Dummy()
        return x

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
    # Acceleration logs (may be missing in some runs)
    acc = np.array(recorder['state_ddot_all']) if 'state_ddot_all' in recorder.results else np.array([])

    # 1. Jerk
    # Check for empty or too short trajectories to avoid errors
    if len(time) < 2:
        return {"jerk": 0.0, "pitch": 0.0, "settling_time": 0.0, "rms_acceleration": 0.0}

    jerk = np.mean(np.abs(np.diff(actions, n=2, axis=0) / np.diff(time[:-1])[:, np.newaxis]))

    # 2. Pitch
    pitch_mean = np.mean(np.abs(states[:, 6])) # 6 is the index for theta (pitch)

    # 2-1. RMS acceleration (comfort indicator)
    def estimate_acc_from_states():
        if len(states) >= 3:
            dt = np.mean(np.diff(time))
            dt = dt if dt > 1e-8 else 1.0
            acc_est = np.diff(states, n=2, axis=0) / (dt * dt)  # shape (T-2, state_dim)
            acc_norm_est = np.linalg.norm(acc_est, axis=1)
            return float(np.sqrt(np.mean(acc_norm_est ** 2)))
        return 0.0

    if acc.size > 0 and not np.allclose(acc, 0.0, atol=1e-6):
        acc_norm = np.linalg.norm(acc, axis=1)
        rms_acc = float(np.sqrt(np.mean(acc_norm ** 2)))
        # If measured acceleration is essentially zero everywhere (sensor not provided), fallback
        if rms_acc < 1e-8:
            rms_acc = estimate_acc_from_states()
    else:
        rms_acc = estimate_acc_from_states()

    # 3. Settling Time (control-theoretic: final-value band + hold time)
    # ------------------------------------------------------------------
    # Definition:
    #  - final_value: mean of the signal over the last tail_duration seconds
    #  - tolerance band: |x(t) - final_value| <= max(rel_tol*|final_value|, abs_tol)
    #  - settling time: earliest t_s such that the signal stays within the band
    #    for a continuous hold_time interval.
    pitch_angle = states[:, 6]  # 6 is the index for theta (pitch angle)

    tail_duration = 1.0   # seconds used to compute final value
    rel_tol = 0.05        # relative tolerance (5%)
    abs_tol = 1e-4        # absolute floor
    hold_time = 0.3       # seconds the signal must stay in-band

    total_time = float(time[-1]) if len(time) > 0 else 0.0
    if len(time) < 2:
        settling_time = total_time
    else:
        dt = float(np.mean(np.diff(time)))
        dt = dt if dt > 1e-8 else 1e-3  # guard against zero/near-zero dt

        # --- Final value over the tail window ---
        tail_start = total_time - tail_duration
        tail_mask = time >= tail_start
        if not np.any(tail_mask):
            tail_mask = np.ones_like(time, dtype=bool)
        final_value = float(np.mean(pitch_angle[tail_mask]))

        # --- Tolerance band ---
        tol = max(rel_tol * abs(final_value), abs_tol)
        inside = np.abs(pitch_angle - final_value) <= tol

        # --- Hold-time condition ---
        hold_steps = max(1, int(np.round(hold_time / dt)))
        settling_time = total_time  # fallback if never settled
        if len(inside) >= hold_steps:
            # Convolution over boolean to find first run of all True of length hold_steps
            window = np.ones(hold_steps, dtype=np.int32)
            conv = np.convolve(inside.astype(np.int32), window, mode='valid')
            hits = np.where(conv == hold_steps)[0]
            if len(hits) > 0:
                first_hit = int(hits[0])
                settling_time = float(time[min(first_hit, len(time) - 1)])

    return {
        "jerk": float(jerk),
        "pitch": float(pitch_mean),
        "settling_time": float(settling_time),
        "rms_acceleration": float(rms_acc)
    }

def generate_data_parser():
    parser = argparse.ArgumentParser(description="Generate trajectory data using vehicle simulation.")
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of simulation episodes to run.')
    parser.add_argument('--dataset-id', type=str, default='A', help="Dataset ID for organizing output files (e.g., 'A', 'B').")
    parser.add_argument('--dataset-name', type=str, default=None, help='Name for the generated dataset file.')
    return parser

