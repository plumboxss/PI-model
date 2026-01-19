import os
import sys
import pickle
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
try:
    from tqdm import tqdm
except Exception:
    # Fallback when tqdm isn't installed (e.g., minimal environments).
    def tqdm(x, **kwargs):
        return x


# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.simulation_utils import get_trajectory_features

def _sigmoid(x: float) -> float:
    # numerically stable sigmoid for scalar
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-8)
    return v / n

def _canonical_pair(i: int, j: int) -> tuple[int, int]:
    """Return an order-invariant representation of a pair."""
    return (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))

def _sample_pair_with_margin(
    rng: np.random.RandomState,
    num_trajectories: int,
    scores: np.ndarray,
    margin_min: float,
    margin_max: float,
    max_tries: int,
):
    """
    Sample (i, j) such that margin = |score[i] - score[j]| is within [margin_min, margin_max].
    Falls back to random pair after max_tries.
    """
    idx1 = idx2 = None
    last_margin = None
    for _ in range(max_tries):
        i, j = rng.choice(num_trajectories, 2, replace=False)
        d = float(scores[i] - scores[j])
        m = abs(d)
        last_margin = m
        if (m >= margin_min) and (m <= margin_max):
            idx1, idx2 = int(i), int(j)
            break
    if idx1 is None or idx2 is None:
        idx1, idx2 = rng.choice(num_trajectories, 2, replace=False)
    return idx1, idx2, float(last_margin if last_margin is not None else 0.0)

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
    # robust iteration: allow list, dict, etc.
    if isinstance(raw_data, dict):
        iterable = sorted(raw_data.items(), key=lambda x: x[0])
    elif isinstance(raw_data, list):
        iterable = enumerate(raw_data)
    else:
        raise ValueError(f"Unsupported raw_data type: {type(raw_data)}")

    for i, result in iterable:
        # Guard against malformed entries
        if not isinstance(result, dict):
            print(f"Warning: Trajectory {i} is not a dict (type={type(result)}); skipping.")
            continue
        feats = result.get('features', None)
        if feats is not None and isinstance(feats, dict):
            # Check if all features are present
            if all(k in feats for k in feature_names):
                feature_vector = np.array([feats[k] for k in feature_names])
                features_list.append(feature_vector)
                trajectories.append({
                    'observations': result.get('state'),
                    'actions': result.get('action'),
                })
            else:
                print(f"Warning: Trajectory {i} missing features. Keys found: {feats.keys()}")
        else:
            print(f"Warning: Trajectory {i} has no usable 'features'; skipping.")
    
    features_matrix = np.array(features_list)
    print(f"Extracted {len(trajectories)} trajectories with features.")

    # 2. 유의미한 궤적 길이 결정 (settling time 기반)
    # settling time 이후의 데이터는 이미 안정화되어서 의미 없으므로 제거
    print("Determining meaningful trajectory length based on settling times...")
    settling_times = features_matrix[:, feature_names.index('settling_time')]
    
    # 원본 궤적 데이터에서 시간 정보를 가져와 settling time을 타임스텝 인덱스로 변환
    settling_time_indices = []
    def iter_raw():
        if isinstance(raw_data, dict):
            return sorted(raw_data.items(), key=lambda x: x[0])
        elif isinstance(raw_data, list):
            return enumerate(raw_data)
        return []

    for i, result in iter_raw():
        if not isinstance(result, dict):
            continue
        feats = result.get('features', None)
        if feats is not None and isinstance(feats, dict):
            if all(k in feats for k in feature_names):
                time_array = np.array(result.get('time', []))
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

    # 4. 가상 유저 생성 (연속 latent 선호) + 확률론 오라클을 위한 준비
    # - 기존 2그룹(전역 평균으로도 잘 맞는 지름길)이 posterior collapse를 유도할 수 있어
    #   유저 선호를 연속 latent로 만들고(사용자별 w가 연속적으로 변화),
    #   라벨은 Bradley–Terry 확률 모델로 생성한다.
    rng = np.random.RandomState(int(args.seed))
    num_users = int(args.num_users)
    latent_dim = int(args.user_latent_dim)
    feature_dim = features_scaled.shape[1]
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    if latent_dim <= 0:
        raise ValueError("user_latent_dim must be positive")

    # user latent z ~ N(0, I)
    user_z = rng.randn(num_users, latent_dim).astype(np.float32)
    # map z -> w (linear map; fixed random matrix) then normalize
    # center weights to reduce global-mean shortcut
    W_map = rng.randn(feature_dim, latent_dim).astype(np.float32)
    raw_w = (user_z @ W_map.T).astype(np.float32)  # (num_users, feature_dim)
    raw_w = raw_w - raw_w.mean(axis=0, keepdims=True)
    # Optionally mix with a shared global direction to increase "commonality" across users.
    # user_diversity=1.0 => original behavior (max diversity), smaller => more commonality.
    user_diversity = float(getattr(args, "user_diversity", 1.0))
    if not (0.0 < user_diversity <= 1.0):
        raise ValueError("--user_diversity must be in (0, 1]")
    w_global = _normalize(rng.randn(feature_dim).astype(np.float32))
    mixed_w = (1.0 - user_diversity) * w_global[None, :] + user_diversity * raw_w
    user_weights = np.stack([_normalize(mixed_w[i]) for i in range(num_users)], axis=0)
    print(f"Created {num_users} virtual users with continuous latent preferences (z_dim={latent_dim}).")
    if user_diversity < 1.0:
        print(f"  Using user_diversity={user_diversity:g}: user weights are mixed with a shared global direction to increase overlap.")

    # 5. 선호도 데이터셋 생성
    print(f"Generating {args.num_pairs} preference pairs using probabilistic (Bradley–Terry) oracle...")
    
    final_dataset = {
        'observations': [], 'actions': [],
        'observations_2': [], 'actions_2': [],
        'labels': [], 'model_id': []
    }
    if bool(getattr(args, "save_pair_indices", False)) or bool(getattr(args, "anchor_ratio", 0.0) > 0.0):
        # pair indices help diagnose cross-user overlap; safe to include (loader ignores extra keys)
        final_dataset["pair_idx1"] = []
        final_dataset["pair_idx2"] = []
    # 시각화/진단용 메타데이터(데이터셋에는 저장하지 않음)
    record_diag = bool(args.visualize)
    diag_score_deltas = []
    diag_p_values = []
    
    num_trajectories = len(trajectories)
    print(f"All trajectories will be truncated to meaningful length of {meaningful_length} timesteps.")
    print(f"  This removes post-settling data that could cause KL loss collapse.")

    # 유저별 균등 쿼터: 총 num_pairs를 num_users로 나눠 각 유저가 동일한 학습 기회를 가짐
    pairs_per_user = args.num_pairs // num_users
    remainder = args.num_pairs % num_users

    # Precompute scores per user for fast sampling: score_u(traj) = w_u · f(traj)
    # features_scaled: (N_traj, feature_dim)
    scores_by_user = user_weights @ features_scaled.T  # (num_users, N_traj)

    # Shared anchor pairs across users to increase overlap/alignment.
    # When users are very different, purely random per-user pair sampling can create near-zero shared comparisons,
    # making it hard to learn a shared representation while still using z.
    anchor_pairs = []
    anchor_ratio = float(getattr(args, "anchor_ratio", 0.0))
    if anchor_ratio < 0.0 or anchor_ratio > 1.0:
        raise ValueError("--anchor_ratio must be in [0, 1]")
    num_anchor_pairs = int(getattr(args, "num_anchor_pairs", 0))
    if anchor_ratio > 0.0:
        if num_anchor_pairs <= 0:
            raise ValueError("--num_anchor_pairs must be positive when --anchor_ratio > 0")
        # Choose anchor pairs by difficulty under a reference score (global direction).
        ref_scores = (w_global @ features_scaled.T).astype(np.float32)  # (N_traj,)
        a_margin_min = float(getattr(args, "anchor_margin_min", args.margin_min))
        a_margin_max = float(getattr(args, "anchor_margin_max", args.margin_max))
        a_max_tries = max(1, int(getattr(args, "anchor_margin_max_tries", args.margin_max_tries)))
        seen = set()
        # generous budget to collect unique anchors
        budget = max(10 * num_anchor_pairs, 1000)
        for _ in range(budget):
            i, j, _m = _sample_pair_with_margin(
                rng=rng,
                num_trajectories=num_trajectories,
                scores=ref_scores,
                margin_min=a_margin_min,
                margin_max=a_margin_max,
                max_tries=a_max_tries,
            )
            key = _canonical_pair(i, j)
            if key in seen:
                continue
            seen.add(key)
            anchor_pairs.append(key)
            if len(anchor_pairs) >= num_anchor_pairs:
                break
        if len(anchor_pairs) < num_anchor_pairs:
            print(f"⚠️  Warning: Only collected {len(anchor_pairs)}/{num_anchor_pairs} unique anchor pairs. "
                  f"Consider relaxing anchor margins or increasing anchor_margin_max_tries.")
        print(f"Using shared anchor pairs: num_anchor_pairs={len(anchor_pairs)}, anchor_ratio={anchor_ratio:g}")

    margin_attempted = 0
    margin_accepted = 0
    sampled_margins = []

    for user_id in tqdm(range(num_users), desc="Generating pairs per user"):
        quota = pairs_per_user + (1 if user_id < remainder else 0)  # 나머지를 앞쪽 유저에게 1개씩 배분
        user_scores = scores_by_user[user_id]  # (N_traj,)

        # a0) shared anchors (overlap across users)
        num_anchor_for_user = int(np.floor(anchor_ratio * float(quota))) if anchor_ratio > 0.0 else 0
        num_remaining = quota - num_anchor_for_user

        if num_anchor_for_user > 0 and len(anchor_pairs) > 0:
            # sample anchors with replacement if needed (keeps overlap high even if quota > anchors)
            anchor_idx = rng.choice(len(anchor_pairs), size=num_anchor_for_user, replace=(num_anchor_for_user > len(anchor_pairs)))
            for ai in anchor_idx:
                idx1, idx2 = anchor_pairs[int(ai)]
                score1 = float(user_scores[idx1])
                score2 = float(user_scores[idx2])
                temperature = max(1e-6, float(args.pair_temperature))
                p = _sigmoid((score1 - score2) / temperature)
                label = 1.0 if (rng.rand() < p) else 0.0
                traj1, traj2 = trajectories[idx1], trajectories[idx2]

                if record_diag:
                    diag_score_deltas.append(float(score1 - score2))
                    diag_p_values.append(float(p))

                final_dataset['observations'].append(traj1['observations'][:meaningful_length])
                final_dataset['actions'].append(traj1['actions'][:meaningful_length])
                final_dataset['observations_2'].append(traj2['observations'][:meaningful_length])
                final_dataset['actions_2'].append(traj2['actions'][:meaningful_length])
                final_dataset['labels'].append(label)
                final_dataset['model_id'].append(user_id)
                if "pair_idx1" in final_dataset:
                    final_dataset["pair_idx1"].append(int(idx1))
                    final_dataset["pair_idx2"].append(int(idx2))

        for _ in range(num_remaining):
            # a) margin 샘플링 비율만큼은 "너무 쉬운 pair"를 피하기 위해 |Δscore| 범위를 맞춘다.
            use_margin = (float(args.margin_sampling_ratio) > 0.0) and (rng.rand() < float(args.margin_sampling_ratio))
            if use_margin:
                margin_attempted += 1
                idx1, idx2, last_m = _sample_pair_with_margin(
                    rng=rng,
                    num_trajectories=num_trajectories,
                    scores=user_scores,
                    margin_min=float(args.margin_min),
                    margin_max=float(args.margin_max),
                    max_tries=max(1, int(args.margin_max_tries)),
                )
                # last_m은 마지막 시도 margin (성공/실패 모두 포함)이라 대략적인 통계용
                if last_m is not None:
                    sampled_margins.append(float(last_m))
                # 성공 여부는 실제 선택된 pair의 margin으로 판단
                d_sel = float(user_scores[idx1] - user_scores[idx2])
                m_sel = abs(d_sel)
                if (m_sel >= float(args.margin_min)) and (m_sel <= float(args.margin_max)):
                    margin_accepted += 1
            else:
                idx1, idx2 = rng.choice(num_trajectories, 2, replace=False)

            # b) 점수 계산
            score1 = float(user_scores[idx1])
            score2 = float(user_scores[idx2])

            # c) 확률론 라벨(Bradley–Terry): p(y=1) = sigmoid((s1 - s2) / T)
            #    T가 작을수록 결정론에 가까워지고, T가 클수록 노이즈가 커진다.
            temperature = max(1e-6, float(args.pair_temperature))
            p = _sigmoid((score1 - score2) / temperature)
            label = 1.0 if (rng.rand() < p) else 0.0
            traj1, traj2 = trajectories[idx1], trajectories[idx2]

            if record_diag:
                diag_score_deltas.append(float(score1 - score2))
                diag_p_values.append(float(p))

            # d. 데이터셋에 추가 (유의미한 길이로 통일, settling time 이후 데이터 제거)
            final_dataset['observations'].append(traj1['observations'][:meaningful_length])
            final_dataset['actions'].append(traj1['actions'][:meaningful_length])
            
            final_dataset['observations_2'].append(traj2['observations'][:meaningful_length])
            final_dataset['actions_2'].append(traj2['actions'][:meaningful_length])

            final_dataset['labels'].append(label)
            final_dataset['model_id'].append(user_id)
            if "pair_idx1" in final_dataset:
                final_dataset["pair_idx1"].append(int(idx1))
                final_dataset["pair_idx2"].append(int(idx2))

    if margin_attempted > 0:
        accept_rate = margin_accepted / float(margin_attempted)
        print(f"Margin sampling attempted={margin_attempted}, accepted={margin_accepted}, accept_rate={accept_rate:.3f}")
        if len(sampled_margins) > 0:
            m_arr = np.array(sampled_margins, dtype=np.float32)
            print(f"Sampled margin stats (last-try): mean={m_arr.mean():.3f}, p50={np.median(m_arr):.3f}, p90={np.percentile(m_arr,90):.3f}")

    # 6. 최종 데이터셋 형식 변환 및 저장
    print("Converting to final format and saving...")
    for k in ['observations', 'actions', 'observations_2', 'actions_2']:
        final_dataset[k] = np.expand_dims(np.array(final_dataset[k]), axis=1)
    
    final_dataset['labels'] = np.expand_dims(np.array(final_dataset['labels']), axis=1)
    final_dataset['model_id'] = np.array(final_dataset['model_id'])
    if "pair_idx1" in final_dataset:
        final_dataset["pair_idx1"] = np.array(final_dataset["pair_idx1"], dtype=np.int32)
        final_dataset["pair_idx2"] = np.array(final_dataset["pair_idx2"], dtype=np.int32)

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
        from src.utils.visualization import (
            plot_preference_distribution,
            plot_preference_oracle_diagnostics,
            plot_user_weight_heatmap,
            plot_pair_overlap_diagnostics,
        )
        
        viz_dir = os.path.join(os.path.dirname(args.output_path), 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        plot_preference_distribution(
            final_dataset,
            save_path=os.path.join(viz_dir, 'preference_distribution.png')
        )

        # 오라클/난이도 진단 플롯 (Δscore, p(y=1), 유저별 라벨 편향 등)
        plot_preference_oracle_diagnostics(
            labels=np.array(final_dataset["labels"]).reshape(-1),
            model_ids=np.array(final_dataset["model_id"]).reshape(-1),
            score_deltas=np.array(diag_score_deltas, dtype=np.float32),
            p_values=np.array(diag_p_values, dtype=np.float32),
            pair_temperature=float(args.pair_temperature),
            margin_min=float(args.margin_min) if float(args.margin_sampling_ratio) > 0.0 else None,
            margin_max=float(args.margin_max) if float(args.margin_sampling_ratio) > 0.0 else None,
            save_path=os.path.join(viz_dir, "preference_oracle_diagnostics.png"),
        )

        # 유저 선호 다양성(연속 latent에서 생성된 w_user) 히트맵
        plot_user_weight_heatmap(
            user_weights=user_weights,
            feature_names=feature_names,
            save_path=os.path.join(viz_dir, "user_weight_heatmap.png"),
        )

        # Cross-user overlap diagnostics (requires pair indices)
        if "pair_idx1" in final_dataset:
            plot_pair_overlap_diagnostics(
                pair_idx1=np.array(final_dataset["pair_idx1"]).reshape(-1),
                pair_idx2=np.array(final_dataset["pair_idx2"]).reshape(-1),
                model_ids=np.array(final_dataset["model_id"]).reshape(-1),
                save_path=os.path.join(viz_dir, "pair_overlap_diagnostics.png"),
            )
        
        print(f"Visualizations saved to {viz_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build preference dataset from raw trajectories using feature-weighted oracle.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the raw trajectory data pkl file from generate_data.py.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the final preference dataset pkl file.')
    parser.add_argument('--num_pairs', type=int, default=20000, help='Number of preference pairs to generate.')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed for user/pair generation.')
    # Continuous user preference latent
    parser.add_argument('--num_users', type=int, default=200, help='Number of virtual users (oracle identities).')
    parser.add_argument('--user_latent_dim', type=int, default=8, help='Latent dim for continuous user preferences.')
    parser.add_argument('--user_diversity', type=float, default=1.0,
                        help='(0,1] Mix user weights with a shared global direction. Smaller => more commonality across users.')
    parser.add_argument('--save_pair_indices', action='store_true',
                        help='Store trajectory index pairs (pair_idx1/pair_idx2) in the dataset for overlap diagnostics.')
    # Probabilistic (Bradley–Terry) labeling
    parser.add_argument('--pair_temperature', type=float, default=0.5,
                        help='Temperature for probabilistic labeling: p=σ((s1-s2)/T). Smaller => more deterministic.')
    # Margin-based pair sampling (reduce easy pairs)
    parser.add_argument('--margin_sampling_ratio', type=float, default=0.9,
                        help='Fraction of pairs sampled with margin constraints (0.0 disables margin sampling).')
    parser.add_argument('--margin_min', type=float, default=0.2, help='Minimum |score1-score2| for margin sampling.')
    parser.add_argument('--margin_max', type=float, default=1.5, help='Maximum |score1-score2| for margin sampling.')
    parser.add_argument('--margin_max_tries', type=int, default=80,
                        help='Max attempts to find a margin-satisfying pair before falling back to random.')
    # Shared anchor pairs across users (increase overlap)
    parser.add_argument('--anchor_ratio', type=float, default=0.0,
                        help='Fraction of each user quota drawn from shared anchor pairs (0 disables).')
    parser.add_argument('--num_anchor_pairs', type=int, default=512,
                        help='Number of unique anchor pairs shared across users (used when anchor_ratio>0).')
    parser.add_argument('--anchor_margin_min', type=float, default=None,
                        help='Anchor pair sampling margin_min under reference score (defaults to margin_min).')
    parser.add_argument('--anchor_margin_max', type=float, default=None,
                        help='Anchor pair sampling margin_max under reference score (defaults to margin_max).')
    parser.add_argument('--anchor_margin_max_tries', type=int, default=None,
                        help='Anchor pair sampling max tries (defaults to margin_max_tries).')
    args = parser.parse_args()
    # Fill anchor margin defaults if None
    if args.anchor_margin_min is None:
        args.anchor_margin_min = args.margin_min
    if args.anchor_margin_max is None:
        args.anchor_margin_max = args.margin_max
    if args.anchor_margin_max_tries is None:
        args.anchor_margin_max_tries = args.margin_max_tries
    main(args)
