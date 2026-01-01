import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# 프로젝트 루트를 Python 경로에 추`가
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
    # src/utils/simulation_utils.py에서 반환하는 키: "jerk", "pitch", "settling_time"
    feature_names = ['jerk', 'pitch', 'settling_time']
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

    # 2. 피처 정규화 및 K-Means 클러스터링
    print(f"Normalizing features and performing K-Means clustering with K={args.num_clusters}...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_matrix)
    
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    centroids_scaled = kmeans.cluster_centers_
    print(f"Clustering complete. Found {len(np.unique(cluster_labels))} clusters.")

    # 3. 각 클러스터의 선호도 방향 자동 정의
    # 전체 피처의 평균(정규화되었으므로 0)을 기준으로 각 클러스터의 특성을 파악합니다.
    # 중심점의 값이 양수이면 해당 피처 값이 평균보다 높은 그룹, 음수이면 낮은 그룹입니다.
    # 이를 바탕으로 각 클러스터에 대한 고유한 선호도 방향 벡터를 생성합니다.
    # 예: "저크는 낮고(-1) 피치는 높은(+1) 것을 선호"하는 그룹 등
    cluster_preference_directions = {}
    for i in range(args.num_clusters):
        # 중심점 값을 부호로 변환하여 선호도 방향으로 사용 (+1: 높을수록 선호, -1: 낮을수록 선호)
        # 예를 들어, 어떤 클러스터의 avg_jerk 중심값이 음수이면, 그 그룹은 낮은 avg_jerk를 선호한다고 해석
        preference_vector = np.sign(centroids_scaled[i])
        # 모든 피처를 낮을수록 선호하는 기본 규칙과 조합하여 다양성 부여
        # 여기서는 간단하게 중심점의 부호를 그대로 사용. 0인 경우 -1로 설정(낮을수록 좋다고 가정)
        preference_vector[preference_vector == 0] = -1 
        cluster_preference_directions[i] = preference_vector
        print(f"Cluster {i} preference directions: {preference_vector}")

    # 각 클러스터의 선호도 점수 계산 함수
    def get_preference_score(features, directions):
        return np.dot(features, directions)

    # 4. 선호도 데이터셋 생성
    print(f"Generating {args.num_pairs} preference pairs for {args.num_clusters} user groups...")
    
    final_dataset = {
        'observations': [], 'actions': [],
        'observations_2': [], 'actions_2': [],
        'labels': [], 'model_id': []
    }
    
    num_trajectories = len(trajectories)
    min_len = min(len(t['observations']) for t in trajectories)
    print(f"All trajectories will be truncated to the minimum length of {min_len}.")

    for _ in tqdm(range(args.num_pairs)):
        # a. 무작위로 두 개의 서로 다른 궤적 선택
        idx1, idx2 = np.random.choice(num_trajectories, 2, replace=False)
        
        # b. 첫 번째 궤적이 속한 클러스터(사용자 그룹)를 기준으로 삼음
        cluster_id = cluster_labels[idx1]
        
        # c. 해당 사용자 그룹의 고유한 선호도 규칙을 가져옴
        preference_directions = cluster_preference_directions[cluster_id]
        
        # d. 해당 그룹의 선호도에 따라 점수 계산
        score1 = get_preference_score(features_scaled[idx1], preference_directions)
        score2 = get_preference_score(features_scaled[idx2], preference_directions)

        # e. 점수가 더 높은 쪽이 선호됨
        # 점수가 높은 쪽을 traj1으로 고정하고 label을 1.0으로 통일
        if score1 >= score2:
            pref_traj, non_pref_traj = trajectories[idx1], trajectories[idx2]
        else:
            pref_traj, non_pref_traj = trajectories[idx2], trajectories[idx1]
        label = 1.0

        # f. 데이터셋에 추가 (궤적 길이 통일)
        final_dataset['observations'].append(pref_traj['observations'][:min_len])
        final_dataset['actions'].append(pref_traj['actions'][:min_len])
        
        final_dataset['observations_2'].append(non_pref_traj['observations'][:min_len])
        final_dataset['actions_2'].append(non_pref_traj['actions'][:min_len])

        final_dataset['labels'].append(label)
        final_dataset['model_id'].append(cluster_id)

    # 5. 최종 데이터셋 형식 변환 및 저장
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
        from src.utils.visualization import plot_clustering_results, plot_preference_distribution
        
        viz_dir = os.path.join(os.path.dirname(args.output_path), 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        plot_clustering_results(
            features_scaled, cluster_labels, centroids_scaled,
            save_path=os.path.join(viz_dir, 'clustering_results.png')
        )
        plot_preference_distribution(
            final_dataset,
            save_path=os.path.join(viz_dir, 'preference_distribution.png')
        )
        
        print(f"Visualizations saved to {viz_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build preference dataset from raw trajectories using feature-based clustering.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the raw trajectory data pkl file from generate_data.py.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the final preference dataset pkl file.')
    parser.add_argument('--num_clusters', type=int, default=16, help='Number of user groups to cluster trajectories into.')
    parser.add_argument('--num_pairs', type=int, default=20000, help='Number of preference pairs to generate.')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    args = parser.parse_args()
    main(args)
