import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm
import os
import sys
from functools import partial
from sklearn.preprocessing import StandardScaler

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.vae import VAEModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_state_rewards(states_b, actions_b, z, comparison_set_C_obs, comparison_set_C_act, vae_model):
    """
    상태(state)들의 보상을 계산하는 함수
    r(s,z) = 1/|C| * sum_{s' in C} P(s > s' | z)
    여기서 s는 상태, z는 잠재 변수, C는 비교 세트
    P(s > s' | z)는 상태 s가 상태 s'보다 더 좋은 확률
    """
    with torch.no_grad():
        T = states_b.shape[0] # 배치의 상태 수
        N = comparison_set_C_obs.shape[0] # 비교 세트의 크기
        
        # 배치 계산을 위한 텐서 준비
        # C의 각 요소와 비교하기 위해 상태 반복: (T, N, D_obs)
        states_b_rpt = states_b.unsqueeze(1).repeat(1, N, 1)  # (T, N, obs_dim)
        actions_b_rpt = actions_b.unsqueeze(1).repeat(1, N, 1)  # (T, N, act_dim)
        # 배치의 각 상태에 대해 C 반복: (T, N, D_obs), (T, N, D_act)
        C_obs_rpt = comparison_set_C_obs.unsqueeze(0).repeat(T, 1, 1)  # (T, N, obs_dim)
        C_act_rpt = comparison_set_C_act.unsqueeze(0).repeat(T, 1, 1)  # (T, N, act_dim)
        
        # z를 (T*N, z_dim)으로 확장
        z_rpt = z.unsqueeze(0).expand(T * N, -1)  # (T*N, z_dim)

        # 모델 입력을 위해 reshape: (T*N, obs_dim) -> (T*N, 1, obs_dim)
        states_reshaped = states_b_rpt.view(T * N, -1).unsqueeze(1)  # (T*N, 1, obs_dim)
        actions_reshaped = actions_b_rpt.view(T * N, -1).unsqueeze(1)  # (T*N, 1, act_dim)
        C_obs_reshaped = C_obs_rpt.view(T * N, -1).unsqueeze(1)  # (T*N, 1, obs_dim)
        C_act_reshaped = C_act_rpt.view(T * N, -1).unsqueeze(1)  # (T*N, 1, act_dim)

        # VAE 모델을 사용하여 상태와 비교 세트의 보상을 계산
        r_states = vae_model.decode_reward(states_reshaped, actions_reshaped, z_rpt)  # (T*N, 1, 1)
        r_C = vae_model.decode_reward(C_obs_reshaped, C_act_reshaped, z_rpt)  # (T*N, 1, 1)
        
        # (T*N, 1, 1) -> (T*N,)로 squeeze
        r_states = r_states.squeeze()  # (T*N,)
        r_C = r_C.squeeze()  # (T*N,)

        # 선호도 확률 P(s > s' | z) 계산
        probs_flat = torch.sigmoid(r_states - r_C)  # (T*N,)
        
        # (T, N) 형태로 변환하여 N에 대해 평균 계산
        probs = probs_flat.view(T, N)  # (T, N)
        
        # 각 상태의 정규화된 보상은 이러한 확률의 평균
        state_rewards = probs.mean(dim=1) # Shape: (T,)
    
    return state_rewards

def create_trajectory_scorer(z_current, vae_model, comparison_set_C_obs, comparison_set_C_act):
    """
    궤적(trajectory)의 점수를 계산하는 함수를 생성
    궤적의 상태들의 평균 정규화된 보상을 기반으로 함
    R(sigma, z) = 1/|sigma| * sum_{s in sigma} r(s,z)
    여기서 sigma는 궤적, z는 잠재 변수, r(s,z)는 상태 s에 대한 보상
    """
    device = next(vae_model.parameters()).device

    def trajectory_scorer(trajectory):
        # 관측값(observations)과 행동(actions) 추출 및 장치로 이동
        obs_np = trajectory['observations']
        act_np = trajectory['actions']
        obs_tensor = torch.from_numpy(obs_np).float().to(device)
        act_tensor = torch.from_numpy(act_np).float().to(device)

        # 궤적의 모든 상태에 대해 정규화된 보상 계산
        state_rewards = calculate_state_rewards(obs_tensor, act_tensor, z_current, comparison_set_C_obs, comparison_set_C_act, vae_model)
        
        # 궤적의 점수는 상태 보상의 평균
        return state_rewards.mean().item()

    return trajectory_scorer


def find_implicit_pair(input_traj, input_traj_features, z_current, vae_model,
                       trajectories, features_matrix, epsilon=0.1, search_sample_size=500):
    """
    현재 z를 기준으로, 가장 정보량이 많은 암시적 궤적을 찾아 선호 쌍을 형성합니다.
    이 함수는 다음 두 가지 조건을 만족하는 후보 궤적을 찾습니다:
    1. 다양성 조건: 입력 궤적과 평균 Jerk 차이가 epsilon 이상이어야 합니다.
    2. 불확실성 조건: P(후보 ≻ 입력 | z)가 0.5에 가장 가까워야 합니다.

    Args:
        input_traj (dict): 사용자가 제공한 입력 궤적.
        input_traj_features (np.array): 입력 궤적의 특징 벡터.
        z_current (torch.Tensor): 현재 잠재 벡터.
        vae_model (VAEModel): 사전 훈련된 VAE 모델.
        trajectories (list): 전체 궤적 데이터셋.
        features_matrix (np.array): 전체 궤적에 대한 특징 매트릭스.
        epsilon (float): 다양성 필터링을 위한 최소 특징 차이 임계값.
        search_sample_size (int): 검색 효율을 위해 탐색할 궤적의 샘플 크기.

    Returns:
        (int, dict): (best_candidate_idx, trajectory dict)
    """
    device = next(vae_model.parameters()).device
    
    # 입력 궤적 준비
    obs_in_np = input_traj['observations']
    act_in_np = input_traj['actions']
    obs_in = torch.from_numpy(obs_in_np).float().to(device)
    act_in = torch.from_numpy(act_in_np).float().to(device)

    # 검색을 위한 무작위 하위 집합 선택
    search_indices = np.random.choice(len(trajectories), search_sample_size, replace=False)
    
    best_candidate_idx = -1
    min_uncertainty_score = float('inf')

    with torch.no_grad():
        # 모든 후보 궤적에 대해 반복
        for idx in tqdm(search_indices, desc="Finding implicit pair", leave=False):
            candidate_traj = trajectories[idx]
            candidate_features = features_matrix[idx]
            
            # 조건 1: 다양성 확보 (평균 Jerk 차이)
            diversity_score = np.abs(input_traj_features[0] - candidate_features[0])
            if diversity_score < epsilon:
                continue
                
            # 후보 궤적 준비
            obs_cand_np = candidate_traj['observations']
            act_cand_np = candidate_traj['actions']

            # 궤적 길이를 동일하게 맞춤 (짧은 쪽 기준)
            min_len = min(obs_in.shape[0], obs_cand_np.shape[0])
            obs_in_trunc = obs_in[:min_len]
            act_in_trunc = act_in[:min_len]
            obs_cand = torch.from_numpy(obs_cand_np[:min_len]).float().to(device)
            act_cand = torch.from_numpy(act_cand_np[:min_len]).float().to(device)

            # z를 궤적 길이에 맞게 확장
            z_expanded = z_current.repeat(min_len, 1)

            # 조건 2: 불확실성이 가장 높은 쌍 선택 (P ≈ 0.5)
            # 각 궤적의 총 보상 계산
            # obs_in_trunc: (min_len, obs_dim) -> (1, min_len, obs_dim)
            obs_in_batch = obs_in_trunc.unsqueeze(0)  # (1, min_len, obs_dim)
            act_in_batch = act_in_trunc.unsqueeze(0)  # (1, min_len, act_dim)
            obs_cand_batch = obs_cand.unsqueeze(0)  # (1, min_len, obs_dim)
            act_cand_batch = act_cand.unsqueeze(0)  # (1, min_len, act_dim)
            z_batch = z_current.unsqueeze(0)  # (1, latent_dim)
            
            r_in = vae_model.decode_reward(obs_in_batch, act_in_batch, z_batch).sum()
            r_cand = vae_model.decode_reward(obs_cand_batch, act_cand_batch, z_batch).sum()
            
            # 선호도 확률 계산: P(후보 > 입력 | z)
            prob = torch.sigmoid(r_cand - r_in).item()
            
            uncertainty_score = abs(prob - 0.5)
            
            if uncertainty_score < min_uncertainty_score:
                min_uncertainty_score = uncertainty_score
                best_candidate_idx = idx

    if best_candidate_idx == -1:
        print("경고: 충분히 다양한 궤적을 찾을 수 없습니다. 무작위 궤적을 반환합니다.")
        random_idx = np.random.choice(len(trajectories))
        return random_idx, trajectories[random_idx]
        
    return best_candidate_idx, trajectories[best_candidate_idx]


class AdaptationLoop:
    def __init__(self, args):
        self.args = args
        
        # 1. 사전 훈련된 VAE 모델 로드
        print("Loading pretrained VAE model...")
        self.vae_model = torch.load(args.vae_model_path)
        self.vae_model.eval()
        self.device = next(self.vae_model.parameters()).device
        print("Model loaded successfully.")

        # 2. 전체 궤적 데이터셋 로드 및 고정 비교 세트 C 생성
        print("Loading trajectory dataset...")
        with open(args.trajectory_dataset_path, 'rb') as f:
            raw_data = pickle.load(f)

        self.trajectories = []
        self.features_list = []
        # 메모리 효율성을 위해 각 궤적의 길이만 저장
        trajectory_lengths = []
        for i in sorted(raw_data.keys()):
            res = raw_data[i]
            if res.get('features'):
                self.trajectories.append({'observations': res['state'], 'actions': res['action']})
                # IMPORTANT: Keep feature order consistent with training oracle
                feats = res['features']
                self.features_list.append(np.array([
                    feats['jerk'],
                    feats['pitch'],
                    feats['settling_time'],
                    feats['rms_acceleration'],
                ], dtype=np.float32))
                trajectory_lengths.append(len(res['state']))

        self.features_matrix = np.array(self.features_list)
        print(f"Loaded {len(self.trajectories)} trajectories.")

        # 2-1. Build training-style oracle for simulated user feedback (two-group assumption)
        # Match build_preference_dataset.py feature order: [jerk, pitch, settling_time, rms_acceleration]
        self._feature_names = ['jerk', 'pitch', 'settling_time', 'rms_acceleration']
        self._scaler = StandardScaler()
        self.features_scaled = self._scaler.fit_transform(self.features_matrix)

        # Two prototype user groups (same intent as training-time groups)
        # Group A: dislikes jerk & rms_acceleration strongly
        wA = np.array([-1.0, 0.0, 0.0, -1.0], dtype=np.float32)
        # Group B: prefers pitch & settling_time strongly
        wB = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        wA = wA / (np.linalg.norm(wA) + 1e-8)
        wB = wB / (np.linalg.norm(wB) + 1e-8)

        if args.oracle_group.upper() == 'A':
            self.oracle_w = wA
        elif args.oracle_group.upper() == 'B':
            self.oracle_w = wB
        else:
            raise ValueError(f"Unsupported oracle_group: {args.oracle_group}. Use 'A' or 'B'.")
        print(f"Using simulated user oracle group: {args.oracle_group.upper()} (feature space)")

        # 메모리 효율적인 방법으로 고정 비교 세트 C 생성
        # 전체를 concatenate하지 않고, 궤적 인덱스와 타임스텝 인덱스를 샘플링
        print(f"Creating fixed comparison set C with size {args.comparison_set_size}...")
        
        # 각 궤적에서 샘플링할 인덱스 생성
        total_states = sum(trajectory_lengths)
        if args.comparison_set_size > total_states:
            print(f"Warning: comparison_set_size ({args.comparison_set_size}) is larger than total states ({total_states}). Using all states.")
            args.comparison_set_size = total_states
        
        # 누적 길이를 사용하여 각 샘플이 어느 궤적에 속하는지 빠르게 찾기
        cumsum_lengths = np.cumsum([0] + trajectory_lengths)
        
        # 무작위로 샘플링할 전역 인덱스 선택
        global_indices = np.random.choice(total_states, args.comparison_set_size, replace=False)
        
        # 각 전역 인덱스를 (trajectory_idx, timestep_idx)로 변환
        sampled_obs = []
        sampled_act = []
        for global_idx in global_indices:
            # 이진 검색으로 궤적 인덱스 찾기
            traj_idx = np.searchsorted(cumsum_lengths, global_idx + 1) - 1
            timestep_idx = global_idx - cumsum_lengths[traj_idx]
            sampled_obs.append(self.trajectories[traj_idx]['observations'][timestep_idx])
            sampled_act.append(self.trajectories[traj_idx]['actions'][timestep_idx])
        
        # 샘플링된 상태와 행동을 텐서로 변환
        self.comparison_set_C_obs = torch.from_numpy(np.array(sampled_obs)).float().to(self.device)
        self.comparison_set_C_act = torch.from_numpy(np.array(sampled_act)).float().to(self.device)
        print(f"Comparison set C created with {len(self.comparison_set_C_obs)} states.")

        # 3. z 및 컨텍스트 초기화
        self.z_current = torch.randn(1, self.vae_model.latent_dim, device=self.device)
        self.context = [] # List to store (traj1_obs, traj2_obs, label)
        self.z_history = [self.z_current.clone()]  # z 변화 추적
        print(f"Initialized z with shape: {self.z_current.shape}")


    def step(self, input_idx):
        """어댑테이션 루프의 한 단계를 수행"""
        print(f"\n--- Step {len(self.context) + 1} ---")
        input_traj = self.trajectories[input_idx]
        input_traj_features = self.features_matrix[input_idx]
        
        # 1. 암시적 쌍 찾기 (이제 trajectory_scorer가 필요 없음)
        print("1. Searching for an implicit pair...")
        implicit_idx, implicit_traj = find_implicit_pair(
            input_traj, input_traj_features, self.z_current, self.vae_model,
            self.trajectories, self.features_matrix,
            epsilon=self.args.diversity_epsilon
        )
        print("Implicit pair found.")

        # 1-1. Simulated user feedback via training-style two-group oracle (pairwise preference)
        score_in = float(np.dot(self.oracle_w, self.features_scaled[input_idx]))
        score_imp = float(np.dot(self.oracle_w, self.features_scaled[implicit_idx]))
        input_label = 1 if score_in >= score_imp else 0  # 1 => input preferred
        print(f"Simulated user preference (oracle group {self.args.oracle_group.upper()}): input_label={input_label}")

        # 2. Update context
        # Order-fixed pair + true label y in {0,1}
        # y=1 => (input_traj ≻ implicit_traj), y=0 => (implicit_traj ≻ input_traj)
        self.context.append((input_traj, implicit_traj, float(input_label)))
        print(f"2. Context updated. Current context size: {len(self.context)}")

        # 3. 전체 컨텍스트를 사용하여 z 재추정
        print("3. Re-inferring z from the updated context...")
        
        # 전체 컨텍스트를 단일 배치로 준비하여 어텐션 인코더에 전달
        trajs1_list, trajs2_list, labels_list = zip(*self.context)
        
        # 컨텍스트에서 최소 궤적 길이를 찾아 패딩/자르기
        min_len_obs1 = min(t['observations'].shape[0] for t in trajs1_list)
        min_len_obs2 = min(t['observations'].shape[0] for t in trajs2_list)
        min_len = min(min_len_obs1, min_len_obs2)

        # 모든 궤적을 최소 길이로 자르고 스택
        obs1_batch = np.array([t['observations'][:min_len] for t in trajs1_list])
        obs2_batch = np.array([t['observations'][:min_len] for t in trajs2_list])
        actions1_batch = np.array([t['actions'][:min_len] for t in trajs1_list])
        actions2_batch = np.array([t['actions'][:min_len] for t in trajs2_list])
        labels_batch = np.array(labels_list)

        # s1/s2는 obs+act를 결합해야 함
        s1_batch = np.concatenate([obs1_batch, actions1_batch], axis=-1)
        s2_batch = np.concatenate([obs2_batch, actions2_batch], axis=-1)

        # 새 모델 인터페이스에 맞게 변환
        # context_s1: (K, T, D_sa) -> (1, K, T, D_sa)
        context_s1 = torch.from_numpy(s1_batch).float().to(self.device).unsqueeze(0)  # (1, K, T, D_sa)
        context_s2 = torch.from_numpy(s2_batch).float().to(self.device).unsqueeze(0)  # (1, K, T, D_sa)
        context_y = torch.from_numpy(labels_batch).float().to(self.device).unsqueeze(-1).unsqueeze(0)  # (1, K, 1)

        with torch.no_grad():
            mean, log_var = self.vae_model.encode_context(context_s1, context_s2, context_y)
            self.z_current = self.vae_model.reparameterization(mean, log_var)
        
        # z 변화 추적
        self.z_history.append(self.z_current.clone())
        
        print(f"z re-inferred successfully. New z mean: {mean.mean().item():.4f}")
        return self.z_current

    def run(self):
        """어댑테이션 세션을 실행"""
        print("\nStarting interactive adaptation loop.")
        print("In each step, a trajectory is selected and a simulated user oracle answers preference based on the chosen user group.")
        
        # 실제 시나리오에서는 외부 입력에 의해 이루어짐.
        # 여기서는 몇 가지 더미 입력을 사용하여 시뮬레이션.
        for i in range(3): # 3 단계의 피드백 시뮬레이션
            # 더미 입력: 데이터셋에서 무작위 궤적 선택
            dummy_input_idx = np.random.randint(len(self.trajectories))
            print(f"\nSimulating user input: Trajectory {dummy_input_idx} (oracle group {self.args.oracle_group.upper()})")
            self.step(dummy_input_idx)
        
        print("\nAdaptation finished.")
        print("Final z (mean of distribution):")
        print(self.z_current)

        # 최종 z 저장
        output_path = self.args.output_z_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(self.z_current, output_path)
        print(f"Final z saved to {output_path}")
        
        # 시각화 생성
        if self.args.visualize:
            print("\nGenerating adaptation visualization plots...")
            from src.utils.visualization import plot_z_evolution, plot_adaptation_summary
            
            viz_dir = os.path.join(os.path.dirname(output_path), 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            plot_z_evolution(self.z_history, save_path=os.path.join(viz_dir, 'z_evolution.png'))
            
            context_sizes = [len(self.context[:i+1]) for i in range(len(self.z_history))]
            plot_adaptation_summary(self.z_history, context_sizes, 
                                  save_path=os.path.join(viz_dir, 'adaptation_summary.png'))
            
            print(f"Visualizations saved to {viz_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="어댑테이션 세션을 실행하여 사용자의 잠재 선호 z를 찾습니다.")
    parser.add_argument('--vae_model_path', type=str, required=True, help='사전 훈련된 VAE 모델 경로 (.pt 파일).')
    parser.add_argument('--trajectory_dataset_path', type=str, required=True, help='궤적 데이터셋 경로 (.pkl 파일).')
    parser.add_argument('--output_z_path', type=str, default='data/adapted_z.pt', help='최종 어댑트된 z 벡터 저장 경로.')
    parser.add_argument('--comparison_set_size', type=int, default=1000, help='고정 비교 세트 C의 상태 수.')
    parser.add_argument('--diversity_epsilon', type=float, default=0.1, help='다양한 쌍을 위한 최소 특성 차이.')
    parser.add_argument('--oracle_group', type=str, default='A', choices=['A', 'B', 'a', 'b'],
                        help="Simulated user group for oracle feedback: 'A' (jerk-hater) or 'B' (pitch-hater).")
    parser.add_argument('--visualize', action='store_true', help='Generate adaptation visualization plots')

    args = parser.parse_args()
    
    # 예제 사용 (학습된 모델과 데이터셋이 필요함):
    # python experiments/run_interactive_adaptation.py --vae_model_path logs/maze2d-twogoals-multimodal-v0/VAE/vae/s0/best_model.pt --trajectory_dataset_path pref_datasets/maze2d-twogoals-multimodal-v0/relabelled_queries_num5000_q1_s16/raw_trajectories.pkl
    
    loop = AdaptationLoop(args)
    loop.run()
