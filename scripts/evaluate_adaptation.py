import argparse
import os
import sys
import pickle
import torch
import numpy as np
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.vae import VAEModel

def get_reward_fn(vae_model, z_adapted):
    """
    적응된 z 벡터를 사용하여 고정된 보상 함수를 반환하는 팩토리 함수.

    Args:
        vae_model (VAEModel): 사전 훈련된 VAE 모델.
        z_adapted (torch.Tensor): 적응 단계를 통해 얻은 최종 z 벡터.

    Returns:
        function: 상태(s)와 행동(a)을 입력받아 스칼라 보상을 반환하는 함수.
    """
    
    device = next(vae_model.parameters()).device
    vae_model.eval() # 평가 모드로 설정

    # z_adapted를 (1, latent_dim) 형태로 유지
    if z_adapted.dim() == 0:
        z_adapted = z_adapted.unsqueeze(0)
    elif z_adapted.dim() > 2:
        z_adapted = z_adapted.mean(dim=[0,1,2]).unsqueeze(0)
    elif z_adapted.dim() == 2 and z_adapted.shape[0] > 1:
        z_adapted = z_adapted.mean(dim=0).unsqueeze(0)
        
    def r_new(s, a):
        """
        새로운 보상 함수 r_new(s, a) = r_φ(s, a, z_adapted).

        Args:
            s (np.array): 단일 상태 벡터. (state_dim,)
            a (np.array): 단일 행동 벡터. (action_dim,)

        Returns:
            float: 계산된 스칼라 보상 값.
        """
        with torch.no_grad():
            # 입력 numpy 배열을 torch 텐서로 변환
            # 단일 timestep이므로 (1, 1, dim) 형태로 변환
            s_tensor = torch.from_numpy(s).float().to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)
            a_tensor = torch.from_numpy(a).float().to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, act_dim)
            
            # 모델의 디코더를 사용하여 보상 계산
            reward = vae_model.decode_reward(s_tensor, a_tensor, z_adapted)  # (1, 1, 1)
            
            return reward.item()

    return r_new

def main(args):
    # 1. 사전 훈련된 VAE 모델과 적응된 z 벡터 로드
    print(f"Loading pretrained VAE model from {args.vae_model_path}...")
    vae_model = torch.load(args.vae_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_model.to(device)
    print("Model loaded successfully.")

    print(f"Loading adapted z vector from {args.adapted_z_path}...")
    z_adapted = torch.load(args.adapted_z_path, map_location=device)
    print(f"Adapted z loaded successfully. Shape: {z_adapted.shape}")

    # 2. 적응된 z를 사용하여 새로운 보상 함수 정의
    reward_function = get_reward_fn(vae_model, z_adapted)
    print("New reward function r_new(s, a) has been defined.")

    # 3. 궤적 데이터셋이 제공되면 평가 및 시각화
    if args.trajectory_dataset_path:
        print(f"\n--- Loading and scoring trajectories from {args.trajectory_dataset_path} ---")
        with open(args.trajectory_dataset_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        # 궤적 점수 계산
        trajectory_scores = {}
        trajectories_list = []
        for i in tqdm(sorted(raw_data.keys()), desc="Scoring trajectories"):
            traj = raw_data[i]
            # 궤적 형식 변환
            if 'state' in traj and 'action' in traj:
                traj_dict = {
                    'observations': traj['state'],
                    'actions': traj['action']
                }
            elif 'observations' in traj and 'actions' in traj:
                traj_dict = traj
            else:
                continue
                
            trajectories_list.append(traj_dict)
            
            total_reward = 0
            for s, a in zip(traj_dict['observations'], traj_dict['actions']):
                total_reward += reward_function(s, a)
            trajectory_scores[i] = total_reward / len(traj_dict['observations'])
        
        # 결과 출력
        sorted_trajectories = sorted(trajectory_scores.items(), key=lambda item: item[1], reverse=True)
        print(f"\nTop 10 trajectories with the highest scores:")
        for traj_id, score in sorted_trajectories[:10]:
            print(f"  Trajectory {traj_id}: Average Reward = {score:.4f}")
        
        # 시각화 생성
        if args.visualize:
            print("\n--- Generating visualization plots ---")
            from src.utils.visualization import plot_reward_distribution, plot_before_after_comparison
            
            viz_dir = os.path.join(os.path.dirname(args.adapted_z_path), 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # 1. 보상 분포
            plot_reward_distribution(
                trajectories_list,
                reward_function,
                top_k=10,
                save_path=os.path.join(viz_dir, 'reward_distribution.png')
            )
            
            # 2. 적응 전/후 비교 (적응 전 보상 함수가 있는 경우)
            if args.before_z_path:
                print("Loading before-adaptation z vector for comparison...")
                z_before = torch.load(args.before_z_path, map_location=device)
                reward_before = get_reward_fn(vae_model, z_before)
                
                plot_before_after_comparison(
                    trajectories_list,
                    reward_before,
                    reward_function,
                    save_path=os.path.join(viz_dir, 'before_after_comparison.png')
                )
            
            print(f"Visualizations saved to {viz_dir}")
    else:
        print("\nNote: Use --trajectory_dataset_path to score and visualize trajectories.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the adapted reward function.")
    parser.add_argument('--vae_model_path', type=str, required=True, help='Path to the pretrained VAE model (.pt file).')
    parser.add_argument('--adapted_z_path', type=str, required=True, help='Path to the adapted z vector file (adapted_z.pt).')
    parser.add_argument('--trajectory_dataset_path', type=str, default=None, help='Path to trajectory dataset for scoring (.pkl file).')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    parser.add_argument('--before_z_path', type=str, default=None, help='Path to before-adaptation z vector for comparison (optional).')
    args = parser.parse_args()
    main(args)


