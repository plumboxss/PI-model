import os
import pickle
import random
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm

from src.utils.simulation_utils import SimulationRecorder, generate_data_parser, get_trajectory_features
from src.simulation.oracle import create_oracle_from_config
from src.simulation.controller import PController
from src.simulation.env import SingleScenarioEnv

import yaml

def load_oracle_config_from_yaml(oracle_name):
    """Helper function to load oracle config name/path if needed, or just return the name."""
    # In this simple setup, we just pass the oracle name (e.g., 'A') to create_oracle_from_config
    # create_oracle_from_config internally constructs the path "configs/oracle_{name}.yaml"
    return oracle_name


# Added from plant.py for feature calculation
STATE_KEYS = ["dz_com", "dtheta", "dz_us_f", "dz_us_r", "dx_com", 
              "z_com", "theta", "z_us_f", "z_us_r", "x_com"]
JERK_KEYS = ["dddz_com", "dddtheta", "dddz_us_f", "dddz_us_r", "dddx_com"]

oracle = None
def init_worker(oracle_config_name):
    global oracle
    oracle_config = load_oracle_config_from_yaml(oracle_config_name)
    oracle = create_oracle_from_config(oracle_config)

def run_single_episode(args):
    global oracle
    eride_p_gain, episode_idx = args
    
    env = SingleScenarioEnv()
    controller = PController(kp=eride_p_gain)
    recorder = SimulationRecorder(env, controller, downsample=20)

    recorder.simulate(seed=None)

    result_for_oracle = {
        "state": recorder['state'],
        "state_ddot": recorder['state_ddot']
    }

    oracle_reward, oracle_probability, oracle_response = oracle(result_for_oracle)
    features = get_trajectory_features(recorder)

    essential_result = {
        "eride_p_gain": eride_p_gain,
        "oracle_reward": oracle_reward,
        "oracle_probability": oracle_probability,
        "oracle_response": oracle_response,
        "features": features,
        "time": recorder['time'],
        "state": recorder['state_all'],
        "state_ddot": recorder['state_ddot_all'],
        "state_dddot": recorder['state_dddot_all'],
        "action": recorder['action_all'],
        "reward": recorder['reward_all']
    }
    return episode_idx, essential_result

def generate_dataset(num_episode, dataset_name, oracle_name):
    args_list = []

    for idx in range(num_episode):
        eride_p_gain = random.randint(30, 300)
        args_list.append((eride_p_gain, idx))

    total_episode = len(args_list)
    print(f"Current available CPU cores: {cpu_count()}")
    print(f"Starting processing with {min(cpu_count(), 8)} workers for {total_episode} episodes...")

    results = {}
    episode_len = 0; processed = 0
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker, initargs=(oracle_name,)) as pool:
        for episode_idx, result in pool.imap_unordered(run_single_episode, args_list):
            new_episode_len = len(result['time'])
            if episode_len == 0:
                episode_len = new_episode_len
            elif episode_len != new_episode_len:
                print(f"Warning: Episode length mismatch! Expected {episode_len}, got {new_episode_len}.")

            results[episode_idx] = result
            processed += 1

            progress = int(processed / total_episode * 100)
            if progress % 10 == 0 and processed % (total_episode // 10) == 0:
                print(f"Processed {processed:,}/{total_episode:,} ({progress}%) episodes.")

    save_path = f"artifacts/{oracle_name}/datasets/{dataset_name}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Dataset saved to {save_path}")

    viz_save_dir = f"artifacts/{oracle_name}/datasets/visualizations/{dataset_name}"
    # The original code had visualize_oracle_data here, but it's not imported.
    # Assuming it's meant to be removed or replaced with a placeholder if needed.
    # For now, commenting out to avoid errors.
    # visualize_oracle_data(save_path, save_dir=viz_save_dir)

# python scripts/generate_data.py --num-episodes 40000 --oracle-name A
def main():
    parser = generate_data_parser()
    args = parser.parse_args()
    num_episodes = args.num_episodes
    oracle_name = args.oracle_name
    dataset_name = f"{num_episodes}" if args.dataset_name is None else args.dataset_name

    print("=" * 50)
    print(f"Generating dataset with {num_episodes} episodes")
    print(f"Dataset name: {dataset_name}")
    print(f"Oracle config: {oracle_name}")
    generate_dataset(num_episodes, dataset_name, oracle_name)
    print("=" * 50)

if __name__ == "__main__":
    main()