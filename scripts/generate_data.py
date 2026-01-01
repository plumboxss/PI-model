import os
import sys
import pickle
import random
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.simulation_utils import SimulationRecorder, generate_data_parser, get_trajectory_features
from src.simulation.controller import PController
from src.simulation.env import SingleScenarioEnv

def run_single_episode(args):
    eride_p_gain, episode_idx = args
    
    env = SingleScenarioEnv()
    controller = PController(kp=eride_p_gain)
    recorder = SimulationRecorder(env, controller, downsample=20)

    recorder.simulate(seed=None)

    features = get_trajectory_features(recorder)

    essential_result = {
        "eride_p_gain": eride_p_gain,
        "features": features,
        "time": recorder['time'],
        "state": recorder['state_all'],
        "state_ddot": recorder['state_ddot_all'],
        "state_dddot": recorder['state_dddot_all'],
        "action": recorder['action_all']
    }
    return episode_idx, essential_result

def generate_dataset(num_episode, dataset_name, dataset_id, visualize=False):
    args_list = []

    for idx in range(num_episode):
        eride_p_gain = random.randint(30, 300)
        args_list.append((eride_p_gain, idx))

    total_episode = len(args_list)
    print(f"Current available CPU cores: {cpu_count()}")
    print(f"Starting processing with {min(cpu_count(), 8)} workers for {total_episode} episodes...")

    results = {}
    episode_len = 0; processed = 0
    with Pool(processes=min(cpu_count(), 8)) as pool:
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

    save_path = f"artifacts/{dataset_id}/datasets/{dataset_name}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Dataset saved to {save_path}")
    
    # 시각화 생성
    if visualize:
        print("\nGenerating data visualization plots...")
        from src.utils.visualization import plot_trajectory_samples, plot_feature_distributions
        
        viz_dir = f"artifacts/{dataset_id}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        plot_trajectory_samples(results, n_samples=6, save_path=os.path.join(viz_dir, 'trajectory_samples.png'))
        plot_feature_distributions(results, save_path=os.path.join(viz_dir, 'feature_distributions.png'))
        
        print(f"Visualizations saved to {viz_dir}")

# python scripts/generate_data.py --num-episodes 40000 --dataset-id A
def main():
    parser = generate_data_parser()
    parser.add_argument('--visualize', action='store_true', help='Generate data visualization plots')
    args = parser.parse_args()
    num_episodes = args.num_episodes
    dataset_id = args.dataset_id
    dataset_name = f"{num_episodes}" if args.dataset_name is None else args.dataset_name

    print("=" * 50)
    print(f"Generating dataset with {num_episodes} episodes")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset ID: {dataset_id}")
    generate_dataset(num_episodes, dataset_name, dataset_id, visualize=args.visualize)

if __name__ == "__main__":
    main()