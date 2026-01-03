import os
import sys
import pickle
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.simulation_utils import get_trajectory_features

def main(args):
    # 1. 원시 궤적 데이터 및 피처 로드
    print(f"Loading raw trajectory data from {args.input_path}...")
    with open(args.input_path, 'rb') as f:
        raw_data = pickle.load(f)

    features_list = []
    trajectories = []
    # 피처 이름은 get_trajectory_features 함수가 반환하는 키와 일치해야 합니다.
    # src/utils/simulation_utils.py에서 반환하는 키: "jerk", "pitch", "settling_time", "rms_acceleration"
    feature_names = ['jerk', 'pitch', 'settling_time', 'rms_acceleration']
    for i in sorted(raw_data.keys()):
        result = raw_data[i]
        if 'features' in result and result['features'] is not None:
            # Check if all features are present
            if all(k in result['features'] for k in feature_names):
                feature_vector = np.array([result['features'][k] for k in feature_names])
                features_list.append(feature_vector)
                trajectories.append({
                    'observations': result['state'],
                    'actions': result['action'],
                })
            else:
                print(f"Warning: Trajectory {i} missing features. Keys found: {result['features'].keys()}")
    
    features_matrix = np.array(features_list)
    print(f"Extracted {len(trajectories)} trajectories with features.")

    # 2. 유의미한 궤적 길이 결정 (settling time 기반)
    # settling time 이후의 데이터는 이미 안정화되어서 의미 없으므로 제거
    print("Determining meaningful trajectory length based on settling times...")
    settling_times = features_matrix[:, feature_names.index('settling_time')]
    
    # 원본 궤적 데이터에서 시간 정보를 가져와 settling time을 타임스텝 인덱스로 변환
    settling_time_indices = []
    for i, result in enumerate(sorted(raw_data.keys())):
        if 'features' in result and result['features'] is not None:
            if all(k in result['features'] for k in feature_names):
                time_array = np.array(result['time'])
                settling_time = settling_times[len(settling_time_indices)]
                
                # settling time에 해당하는 타임스텝 인덱스 찾기
                if len(time_array) > 0:
                    # settling time에 가장 가까운 타임스텝 인덱스 찾기
                    settling_idx = np.argmin(np.abs(time_array - settling_time))
                    # 여유분 추가 (settling time의 1.2배 또는 최소 10 타임스텝)
                    settling_idx = min(int(settling_idx * 1.2) + 10, len(time_array) - 1)
                    settling_time_indices.append(settling_idx)
                else:
                    settling_time_indices.append(0)
    
    if len(settling_time_indices) > 0:
        # 95th percentile을 사용하여 대부분의 궤적이 안정화되는 시점 결정
        meaningful_length = int(np.percentile(settling_time_indices, 95))
        # 최소 길이와 비교하여 안전하게 설정
        min_len = min(len(t['observations']) for t in trajectories)
        meaningful_length = min(meaningful_length, min_len)
        print(f"Meaningful trajectory length determined: {meaningful_length} timesteps (95th percentile of settling times)")
        print(f"  Settling time indices: min={min(settling_time_indices)}, max={max(settling_time_indices)}, mean={np.mean(settling_time_indices):.1f}")
    else:
        # Fallback: 최소 길이 사용
        meaningful_length = min(len(t['observations']) for t in trajectories)
        print(f"Warning: Could not determine meaningful length from settling times. Using minimum length: {meaningful_length}")

    # 3. 피처 정규화 (단위 차이를 제거하여 공정한 가중치 적용)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_matrix)

    # 4. 가상 유저 생성 (feature-weighted oracle)
    num_users = 200
    user_weights = []
    for _ in range(num_users):
        w = np.random.uniform(-1.0, 1.0, size=len(feature_names))
        norm = np.linalg.norm(w) + 1e-8
        user_weights.append(w / norm)
    user_weights = np.array(user_weights)
    print(f"Created {num_users} virtual users (random normalized weights in feature space).")

    # 5. 선호도 데이터셋 생성
    print(f"Generating {args.num_pairs} preference pairs using feature-weighted oracle...")
    
    final_dataset = {
        'observations': [], 'actions': [],
        'observations_2': [], 'actions_2': [],
        'labels': [], 'model_id': []
    }
    
    num_trajectories = len(trajectories)
    print(f"All trajectories will be truncated to meaningful length of {meaningful_length} timesteps.")
    print(f"  This removes post-settling data that could cause KL loss collapse.")

    for _ in tqdm(range(args.num_pairs)):
        # a. 무작위로 두 개의 서로 다른 궤적 선택
        idx1, idx2 = np.random.choice(num_trajectories, 2, replace=False)
        
        # b. 가상 유저 선택 및 점수 계산
        user_id = np.random.randint(0, num_users)
        w = user_weights[user_id]
        score1 = float(np.dot(w, features_scaled[idx1]))
        score2 = float(np.dot(w, features_scaled[idx2]))

        # c. 점수가 더 높은 궤적이 선호됨
        if score1 >= score2:
            pref_traj, non_pref_traj = trajectories[idx1], trajectories[idx2]
        else:
            pref_traj, non_pref_traj = trajectories[idx2], trajectories[idx1]
        label = 1.0

        # f. 데이터셋에 추가 (유의미한 길이로 통일, settling time 이후 데이터 제거)
        final_dataset['observations'].append(pref_traj['observations'][:meaningful_length])
        final_dataset['actions'].append(pref_traj['actions'][:meaningful_length])
        
        final_dataset['observations_2'].append(non_pref_traj['observations'][:meaningful_length])
        final_dataset['actions_2'].append(non_pref_traj['actions'][:meaningful_length])

        final_dataset['labels'].append(label)
        final_dataset['model_id'].append(user_id)

    # 6. 최종 데이터셋 형식 변환 및 저장
    print("Converting to final format and saving...")
    for k in ['observations', 'actions', 'observations_2', 'actions_2']:
        final_dataset[k] = np.expand_dims(np.array(final_dataset[k]), axis=1)
    
    final_dataset['labels'] = np.expand_dims(np.array(final_dataset['labels']), axis=1)
    final_dataset['model_id'] = np.array(final_dataset['model_id'])

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(final_dataset, f)
    
    print(f"Preference dataset saved to {args.output_path}")
    print("Dataset shapes:")
    for k, v in final_dataset.items():
        print(f"  {k}: {v.shape}")
    
    # 시각화 생성
    if args.visualize:
        print("\nGenerating preference dataset visualization plots...")
        from src.utils.visualization import plot_preference_distribution
        
        viz_dir = os.path.join(os.path.dirname(args.output_path), 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        plot_preference_distribution(
            final_dataset,
            save_path=os.path.join(viz_dir, 'preference_distribution.png')
        )
        
        print(f"Visualizations saved to {viz_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build preference dataset from raw trajectories using feature-weighted oracle.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the raw trajectory data pkl file from generate_data.py.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the final preference dataset pkl file.')
    parser.add_argument('--num_pairs', type=int, default=20000, help='Number of preference pairs to generate.')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    args = parser.parse_args()
    main(args)
