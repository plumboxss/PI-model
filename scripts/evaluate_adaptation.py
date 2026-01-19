import argparse
import os
import sys
import pickle
import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.vae import VAEModel
from src.utils.preprocessing import (
    PreprocessStats,
    infer_preprocess_stats_path,
    preprocess_trajectory,
    preprocess_state_action,
)

def get_reward_fn(vae_model, z_adapted, preprocess_stats: PreprocessStats | None = None):
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
            if preprocess_stats is not None:
                s, a = preprocess_state_action(np.asarray(s), np.asarray(a), preprocess_stats)
            # 단일 timestep이므로 (1, 1, dim) 형태로 변환
            s_tensor = torch.from_numpy(np.asarray(s)).float().to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, obs_dim)
            a_tensor = torch.from_numpy(np.asarray(a)).float().to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, act_dim)
            
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

    # Load preprocessing stats for strict train/eval consistency (P0)
    preprocess_stats = None
    stats_path = args.preprocess_stats_path or infer_preprocess_stats_path(args.vae_model_path)
    if os.path.exists(stats_path):
        preprocess_stats = PreprocessStats.from_npz(stats_path)
        print(f"Loaded preprocessing stats from: {stats_path}")
    else:
        print(f"⚠️  Warning: preprocessing_stats.npz not found at '{stats_path}'. "
              f"Evaluation will use raw inputs (may mismatch training).")

    print(f"Loading adapted z vector from {args.adapted_z_path}...")
    z_adapted = torch.load(args.adapted_z_path, map_location=device)
    print(f"Adapted z loaded successfully. Shape: {z_adapted.shape}")

    # 2. 적응된 z를 사용하여 새로운 보상 함수 정의
    reward_function = get_reward_fn(vae_model, z_adapted, preprocess_stats=preprocess_stats)
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

            # Preprocess whole trajectory if stats exist (train-consistent)
            obs_np = np.asarray(traj_dict['observations'])
            act_np = np.asarray(traj_dict['actions'])
            if preprocess_stats is not None:
                obs_np, act_np = preprocess_trajectory(obs_np, act_np, preprocess_stats)
            # Average reward over (possibly downsampled) timesteps
            total_reward = 0.0
            for s, a in zip(obs_np, act_np):
                total_reward += reward_function(s, a)
            trajectory_scores[i] = total_reward / max(1, len(obs_np))
        
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

        # 3-1. (P3) Pairwise preference evaluation with simulated oracle (holdout pairs)
        if args.oracle_group is not None:
            print(f"\n--- Pairwise evaluation (oracle_group={args.oracle_group.upper()}) ---")
            # Build feature matrix from raw_data (same order as training oracle)
            feats_list = []
            traj_map = {}  # idx -> (obs, act)
            keys = []
            for k in sorted(raw_data.keys()):
                res = raw_data[k]
                if isinstance(res, dict) and res.get("features") is not None:
                    f = res["features"]
                    if all(name in f for name in ["jerk", "pitch", "settling_time", "rms_acceleration"]):
                        feats_list.append([f["jerk"], f["pitch"], f["settling_time"], f["rms_acceleration"]])
                        traj_map[len(keys)] = (np.asarray(res["state"]), np.asarray(res["action"]))
                        keys.append(k)
            feats = np.asarray(feats_list, dtype=np.float32)
            scaler = StandardScaler()
            feats_scaled = scaler.fit_transform(feats)

            # Two prototype oracle groups (must match run_interactive_adaptation.py)
            wA = np.array([-1.0, 0.0, 0.0, -1.0], dtype=np.float32)
            wB = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
            wA = wA / (np.linalg.norm(wA) + 1e-8)
            wB = wB / (np.linalg.norm(wB) + 1e-8)
            w = wA if args.oracle_group.upper() == "A" else wB

            n = len(feats_scaled)
            if n < 2:
                print("Not enough trajectories with features to compute pairwise metrics.")
            else:
                rng = np.random.RandomState(args.eval_seed)
                pairs = rng.randint(0, n, size=(args.eval_num_pairs, 2))
                # ensure i != j
                mask_same = pairs[:, 0] == pairs[:, 1]
                while np.any(mask_same):
                    pairs[mask_same, 1] = rng.randint(0, n, size=int(mask_same.sum()))
                    mask_same = pairs[:, 0] == pairs[:, 1]

                y_true = []
                y_score = []
                ret_cache: dict[int, float] = {}

                def get_return(idx: int) -> float:
                    if idx in ret_cache:
                        return ret_cache[idx]
                    obs, act = traj_map[idx]
                    if preprocess_stats is not None:
                        obs, act = preprocess_trajectory(obs, act, preprocess_stats)
                    # torch batch
                    obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)  # (1,T,obs_dim)
                    act_t = torch.from_numpy(act).float().to(device).unsqueeze(0)  # (1,T,act_dim)
                    z = z_adapted if z_adapted.dim() == 2 else z_adapted.unsqueeze(0)
                    with torch.no_grad():
                        r = vae_model.decode_reward(obs_t, act_t, z).sum().item()
                    # Match training scaling
                    ret = r / float(getattr(vae_model, "scaling", 1.0))
                    ret_cache[idx] = ret
                    return ret

                for i_idx, j_idx in pairs:
                    score_i = float(np.dot(w, feats_scaled[i_idx]))
                    score_j = float(np.dot(w, feats_scaled[j_idx]))
                    y = 1.0 if score_i >= score_j else 0.0
                    Ri = get_return(int(i_idx))
                    Rj = get_return(int(j_idx))
                    p = 1.0 / (1.0 + np.exp(-(Ri - Rj)))
                    y_true.append(y)
                    y_score.append(p)

                y_true_arr = np.asarray(y_true, dtype=np.float32)
                y_score_arr = np.asarray(y_score, dtype=np.float32)
                bce = float(-(y_true_arr * np.log(y_score_arr + 1e-8) + (1 - y_true_arr) * np.log(1 - y_score_arr + 1e-8)).mean())
                acc = float(((y_score_arr >= 0.5) == (y_true_arr >= 0.5)).mean())
                try:
                    auc = float(roc_auc_score(y_true_arr, y_score_arr))
                except Exception:
                    auc = float("nan")

                print(f"Pairwise metrics (holdout): BCE={bce:.4f}, Acc={acc:.4f}, AUC={auc:.4f}")
    else:
        print("\nNote: Use --trajectory_dataset_path to score and visualize trajectories.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the adapted reward function.")
    parser.add_argument('--vae_model_path', type=str, required=True, help='Path to the pretrained VAE model (.pt file).')
    parser.add_argument('--adapted_z_path', type=str, required=True, help='Path to the adapted z vector file (adapted_z.pt).')
    parser.add_argument('--trajectory_dataset_path', type=str, default=None, help='Path to trajectory dataset for scoring (.pkl file).')
    parser.add_argument('--preprocess_stats_path', type=str, default=None,
                        help="Path to preprocessing_stats.npz saved during training. "
                             "If omitted, inferred from the vae_model_path directory.")
    parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    parser.add_argument('--before_z_path', type=str, default=None, help='Path to before-adaptation z vector for comparison (optional).')

    # (P3) Pairwise evaluation options (oracle-simulated holdout)
    parser.add_argument('--oracle_group', type=str, default=None, choices=['A', 'B', 'a', 'b'],
                        help="If set, compute pairwise AUC/BCE/Acc on oracle-simulated holdout pairs.")
    parser.add_argument('--eval_num_pairs', type=int, default=5000, help='Number of holdout pairs for pairwise evaluation.')
    parser.add_argument('--eval_seed', type=int, default=0, help='RNG seed for pairwise evaluation sampling.')
    args = parser.parse_args()
    main(args)


