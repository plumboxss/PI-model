"""
프로젝트 전체 성과 확인을 위한 종합 시각화 유틸리티
"""
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용 가능
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional, Callable, Tuple
import os
from pathlib import Path


# ============================================================================
# 1. 데이터 생성 단계 시각화
# ============================================================================

def plot_trajectory_samples(trajectories: Dict, n_samples: int = 6, save_path: Optional[str] = None):
    """생성된 궤적 샘플 시각화"""
    indices = list(trajectories.keys())
    if len(indices) < n_samples:
        n_samples = len(indices)
    sample_indices = np.random.choice(indices, n_samples, replace=False)
    
    n_cols = 2
    n_rows = (n_samples + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, traj_idx in enumerate(sample_indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        traj = trajectories[traj_idx]
        state = traj['state']
        time = np.arange(len(state))
        
        obs_dim = state.shape[1] if len(state.shape) > 1 else 1
        n_dims = min(3, obs_dim)
        
        for i in range(n_dims):
            ax.plot(time, state[:, i], label=f'State[{i}]', alpha=0.7)
        
        if 'features' in traj and traj['features']:
            features = traj['features']
            jerk = features.get('jerk', 'N/A')
            pitch = features.get('pitch', 'N/A')
            title = f'Traj {traj_idx}\nJerk: {jerk:.3f}, Pitch: {pitch:.3f}'
        else:
            title = f'Trajectory {traj_idx}'
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('State Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Sample Trajectories', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_distributions(trajectories: Dict, save_path: Optional[str] = None):
    """궤적 특징 분포 시각화"""
    jerks, pitches, settling_times, rms_accs = [], [], [], []
    
    for traj in trajectories.values():
        if 'features' in traj and traj['features']:
            features = traj['features']
            if 'jerk' in features:
                jerks.append(features['jerk'])
            if 'pitch' in features:
                pitches.append(features['pitch'])
            if 'settling_time' in features:
                settling_times.append(features['settling_time'])
            if 'rms_acceleration' in features:
                rms_accs.append(features['rms_acceleration'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    if jerks:
        axes[0].hist(jerks, bins=30, alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(jerks), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(jerks):.4f}')
        axes[0].set_xlabel('Jerk')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Jerk Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    if pitches:
        axes[1].hist(pitches, bins=30, alpha=0.7, edgecolor='black', color='orange')
        axes[1].axvline(np.mean(pitches), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(pitches):.4f}')
        axes[1].set_xlabel('Pitch')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Pitch Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    if settling_times:
        axes[2].hist(settling_times, bins=30, alpha=0.7, edgecolor='black', color='green')
        axes[2].axvline(np.mean(settling_times), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(settling_times):.4f}')
        axes[2].set_xlabel('Settling Time')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Settling Time Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    if rms_accs:
        axes[3].hist(rms_accs, bins=30, alpha=0.7, edgecolor='black', color='purple')
        axes[3].axvline(np.mean(rms_accs), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rms_accs):.4f}')
        axes[3].set_xlabel('RMS Acceleration')
        axes[3].set_ylabel('Frequency')
        axes[3].set_title('RMS Acceleration Distribution')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].axis('off')
        axes[3].text(0.5, 0.5, 'No RMS Acceleration data', ha='center', va='center', fontsize=10, transform=axes[3].transAxes)
    
    plt.suptitle('Trajectory Feature Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 2. 선호도 데이터셋 구축 단계 시각화
# ============================================================================

def plot_clustering_results(features_matrix: np.ndarray, cluster_labels: np.ndarray, 
                          cluster_centers: np.ndarray, save_path: Optional[str] = None):
    """클러스터링 결과 시각화"""
    from sklearn.decomposition import PCA
    
    # PCA로 2D 투영
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_matrix)
    centers_2d = pca.transform(cluster_centers)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 클러스터 분포
    ax1 = axes[0]
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=f'Cluster {label}', alpha=0.6, s=20)
    
    # 클러스터 중심
    ax1.scatter(centers_2d[:, 0], centers_2d[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroids', zorder=5)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.set_title('Clustering Results (PCA Projection)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. 클러스터 크기
    ax2 = axes[1]
    cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
    ax2.bar(range(len(unique_labels)), cluster_sizes, alpha=0.7, color=colors[:len(unique_labels)])
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Number of Trajectories')
    ax2.set_title('Cluster Sizes')
    ax2.set_xticks(range(len(unique_labels)))
    ax2.set_xticklabels([f'C{i}' for i in unique_labels])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_preference_distribution(dataset: Dict, save_path: Optional[str] = None):
    """선호도 쌍 분포 시각화"""
    labels = dataset['labels']
    model_ids = dataset['model_id']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Label 분포
    ax1 = axes[0]
    unique_labels, counts = np.unique(labels, return_counts=True)
    ax1.bar(unique_labels, counts, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Label')
    ax1.set_ylabel('Count')
    ax1.set_title('Preference Label Distribution')
    ax1.set_xticks(unique_labels)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Model ID 분포
    ax2 = axes[1]
    unique_model_ids, model_counts = np.unique(model_ids, return_counts=True)
    n_users = len(unique_model_ids)
    ax2.set_xlabel('User (model_id)')
    ax2.set_ylabel('Number of Pairs')
    ax2.grid(True, alpha=0.3, axis='y')
    if n_users <= 30:
        ax2.bar(range(n_users), model_counts, alpha=0.7, edgecolor='black')
        ax2.set_title('Preference Pairs per User (by ID)')
        ax2.set_xticks(range(n_users))
        ax2.set_xticklabels([f'G{i}' for i in unique_model_ids], rotation=45)
    else:
        # Too many users: show distribution of counts rather than unreadable per-ID bars
        ax2.hist(model_counts, bins=min(30, max(5, int(np.sqrt(n_users)))), alpha=0.7, edgecolor='black')
        ax2.set_title('Preference Pairs per User (count distribution)')
        ax2.text(
            0.98,
            0.98,
            f"users={n_users}\nmin={int(model_counts.min())}\nmax={int(model_counts.max())}\nmean={model_counts.mean():.1f}",
            transform=ax2.transAxes,
            ha='right',
            va='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        )
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_preference_oracle_diagnostics(
    labels: np.ndarray,
    model_ids: np.ndarray,
    score_deltas: np.ndarray,
    p_values: np.ndarray,
    save_path: Optional[str] = None,
    pair_temperature: Optional[float] = None,
    margin_min: Optional[float] = None,
    margin_max: Optional[float] = None,
):
    """
    오라클 기반 선호 데이터셋 진단 플롯.
    - Δscore 분포(난이도), p(y=1) 분포(노이즈), empirical P(y=1|Δ) (일관성), 유저별 라벨 편향
    """
    labels = np.asarray(labels).reshape(-1)
    model_ids = np.asarray(model_ids).reshape(-1)
    score_deltas = np.asarray(score_deltas).reshape(-1)
    p_values = np.asarray(p_values).reshape(-1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) |Δscore| (margin) distribution
    ax = axes[0, 0]
    abs_d = np.abs(score_deltas)
    ax.hist(abs_d, bins=50, alpha=0.75, edgecolor='black')
    if margin_min is not None:
        ax.axvline(float(margin_min), color='r', linestyle='--', linewidth=2, label=f"margin_min={float(margin_min):.2f}")
    if margin_max is not None:
        ax.axvline(float(margin_max), color='g', linestyle='--', linewidth=2, label=f"margin_max={float(margin_max):.2f}")
    ax.set_title('Pair Difficulty: |Δscore| Distribution')
    ax.set_xlabel('|score1 - score2|')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')

    # 2) p(y=1) distribution
    ax = axes[0, 1]
    ax.hist(p_values, bins=40, alpha=0.75, edgecolor='black', color='orange')
    ax.axvline(float(np.mean(p_values)), color='r', linestyle='--', linewidth=2, label=f"mean={np.mean(p_values):.3f}")
    ax.set_title('Oracle Stochasticity: p(y=1) Distribution')
    ax.set_xlabel('p(y=1)')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')

    # 3) empirical P(y=1 | Δscore) with optional theoretical sigmoid overlay
    ax = axes[1, 0]
    # bin deltas to reduce noise
    n_bins = 25
    lo, hi = np.percentile(score_deltas, [2, 98])
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    emp = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = (score_deltas >= edges[i]) & (score_deltas < edges[i + 1])
        counts[i] = int(m.sum())
        if counts[i] > 20:
            emp[i] = float(labels[m].mean())
    ax.plot(centers, emp, 'o-', label='empirical mean label', alpha=0.9)
    if pair_temperature is not None and pair_temperature > 0:
        T = float(pair_temperature)
        xs = np.linspace(lo, hi, 200)
        ys = 1.0 / (1.0 + np.exp(-xs / T))
        ax.plot(xs, ys, '-', label=f'sigmoid(Δ/T), T={T:g}', alpha=0.8)
    ax.set_title('Consistency: P(y=1 | Δscore)')
    ax.set_xlabel('Δscore = score1 - score2')
    ax.set_ylabel('P(y=1)')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # 4) per-user label mean distribution (bias)
    ax = axes[1, 1]
    unique_users = np.unique(model_ids)
    user_means = []
    user_counts = []
    for u in unique_users:
        m = model_ids == u
        user_counts.append(int(m.sum()))
        if m.sum() > 0:
            user_means.append(float(labels[m].mean()))
    user_means = np.array(user_means, dtype=float)
    ax.hist(user_means, bins=30, alpha=0.75, edgecolor='black', color='green')
    ax.axvline(float(np.mean(user_means)), color='r', linestyle='--', linewidth=2, label=f"mean={np.mean(user_means):.3f}")
    ax.set_title('Per-user Label Bias: mean(y) over users')
    ax.set_xlabel('mean label per user')
    ax.set_ylabel('Number of users')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best')

    plt.suptitle('Preference Dataset Oracle Diagnostics', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pair_overlap_diagnostics(
    pair_idx1: np.ndarray,
    pair_idx2: np.ndarray,
    model_ids: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Cross-user overlap diagnostics.
    Shows how many unique (i,j) pairs are shared across multiple users.
    This is important when user preferences are very different: without shared comparisons,
    learning a shared representation while using z can become unstable.
    """
    pair_idx1 = np.asarray(pair_idx1).reshape(-1)
    pair_idx2 = np.asarray(pair_idx2).reshape(-1)
    model_ids = np.asarray(model_ids).reshape(-1)
    if not (pair_idx1.shape[0] == pair_idx2.shape[0] == model_ids.shape[0]):
        raise ValueError("pair_idx1, pair_idx2, model_ids must have the same length")

    # Canonicalize pairs (order-invariant)
    i = np.minimum(pair_idx1, pair_idx2).astype(np.int64)
    j = np.maximum(pair_idx1, pair_idx2).astype(np.int64)

    # Count in how many distinct users each unique pair appears
    # Approach: build (pair_key, user_id) tuples and unique them, then count per pair_key.
    pair_key = i * (j.max() + 1) + j  # simple hash, safe for visualization
    pair_user = np.stack([pair_key, model_ids.astype(np.int64)], axis=1)
    pair_user_unique = np.unique(pair_user, axis=0)
    unique_pair_keys, user_counts = np.unique(pair_user_unique[:, 0], return_counts=True)

    # Summaries
    total_unique_pairs = int(unique_pair_keys.shape[0])
    shared_ge2 = int((user_counts >= 2).sum())
    shared_ge5 = int((user_counts >= 5).sum())
    shared_ratio = float(shared_ge2 / max(1, total_unique_pairs))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1) Histogram of user counts per unique pair
    ax = axes[0]
    bins = np.arange(1, max(3, int(user_counts.max()) + 2))
    ax.hist(user_counts, bins=bins, alpha=0.75, edgecolor='black')
    ax.set_title('Overlap: #Users per unique pair (i,j)')
    ax.set_xlabel('number of distinct users that saw this pair')
    ax.set_ylabel('number of unique pairs')
    ax.grid(True, alpha=0.3, axis='y')

    # 2) Text summary + bar for shared vs unique
    ax = axes[1]
    counts = np.array([total_unique_pairs - shared_ge2, shared_ge2], dtype=int)
    ax.bar([0, 1], counts, alpha=0.75, edgecolor='black', color=['steelblue', 'orange'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['seen by 1 user', 'seen by >=2 users'])
    ax.set_ylabel('number of unique pairs')
    ax.set_title('Shared vs non-shared pairs')
    ax.grid(True, alpha=0.3, axis='y')
    summary = (
        f"unique_pairs={total_unique_pairs}\n"
        f"shared(>=2 users)={shared_ge2} ({shared_ratio*100:.1f}%)\n"
        f"shared(>=5 users)={shared_ge5}\n"
        f"users={len(np.unique(model_ids))}"
    )
    ax.text(
        0.98, 0.98, summary,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
    )

    plt.suptitle('Pair Overlap Diagnostics', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_user_weight_heatmap(
    user_weights: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    max_users: int = 80,
):
    """유저별 feature-space 가중치(w_user) 히트맵으로 다양성/구조를 빠르게 확인."""
    W = np.asarray(user_weights, dtype=float)
    if W.ndim != 2:
        raise ValueError(f"user_weights must be 2D, got shape={W.shape}")
    n_users, n_feat = W.shape
    if n_feat != len(feature_names):
        # don't crash hard; fallback to generic names
        feature_names = feature_names[:n_feat] if len(feature_names) >= n_feat else [f"f{i}" for i in range(n_feat)]

    if n_users > max_users:
        # sample evenly to keep plot readable
        idx = np.linspace(0, n_users - 1, max_users).astype(int)
        W_plot = W[idx]
        title_users = f"{max_users}/{n_users} users (subsampled)"
    else:
        W_plot = W
        title_users = f"{n_users} users"

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    im = ax.imshow(W_plot, aspect='auto', cmap='coolwarm', vmin=-1.0, vmax=1.0)
    ax.set_title(f'User Preference Weights Heatmap ({title_users})')
    ax.set_xlabel('Feature')
    ax.set_ylabel('User index')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('weight value')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 3. 학습 곡선 시각화
# ============================================================================

def plot_training_curves(metrics_history: Dict[str, List[float]], save_path: Optional[str] = None):
    """학습 곡선 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 어떤 메트릭은 eval_freq 등으로 인해 기록 횟수가 다를 수 있으므로,
    # 제공된 시리즈 중 최대 길이를 epoch 축으로 사용하고 부족한 값은 NaN으로 패딩한다.
    max_len = 0
    for v in metrics_history.values():
        if isinstance(v, list):
            max_len = max(max_len, len(v))
    epochs = range(max_len)
    
    def _padded_series(values: List[float], target_len: int) -> np.ndarray:
        arr = np.array([np.nan if x is None else x for x in values], dtype=float)
        if arr.shape[0] < target_len:
            arr = np.concatenate([arr, np.full(target_len - arr.shape[0], np.nan)])
        return arr
    
    # 1. Loss
    ax1 = axes[0, 0]
    if 'train/loss' in metrics_history:
        ax1.plot(epochs, _padded_series(metrics_history['train/loss'], max_len), label='Train Loss', alpha=0.7)
    if 'eval/loss' in metrics_history:
        ax1.plot(epochs, _padded_series(metrics_history['eval/loss'], max_len), label='Eval Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy
    ax2 = axes[0, 1]
    if 'train/accuracy' in metrics_history:
        ax2.plot(epochs, _padded_series(metrics_history['train/accuracy'], max_len), label='Train Accuracy', alpha=0.7)
    if 'eval/accuracy' in metrics_history:
        ax2.plot(epochs, _padded_series(metrics_history['eval/accuracy'], max_len), label='Eval Accuracy', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. KL Divergence
    ax3 = axes[1, 0]
    if 'train/kld_loss' in metrics_history:
        ax3.plot(epochs, _padded_series(metrics_history['train/kld_loss'], max_len), label='Train KL', alpha=0.7)
    if 'eval/kld_loss' in metrics_history:
        ax3.plot(epochs, _padded_series(metrics_history['eval/kld_loss'], max_len), label='Eval KL', alpha=0.7)
    if 'train/kld_loss_raw' in metrics_history:
        ax3.plot(epochs, _padded_series(metrics_history['train/kld_loss_raw'], max_len), label='Train KL (raw)', alpha=0.7, linestyle='--')
    if 'eval/kld_loss_raw' in metrics_history:
        ax3.plot(epochs, _padded_series(metrics_history['eval/kld_loss_raw'], max_len), label='Eval KL (raw)', alpha=0.7, linestyle='--')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('KL Divergence')
    ax3.set_title('KL Divergence Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. KL Weight (Annealing)
    ax4 = axes[1, 1]
    if 'train/kl_weight' in metrics_history:
        ax4.plot(epochs, _padded_series(metrics_history['train/kl_weight'], max_len), label='KL Weight', alpha=0.7, color='green')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('KL Weight')
    ax4.set_title('KL Weight (Annealing)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Training Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 4. 적응 단계 시각화
# ============================================================================

def plot_z_evolution(z_history: List[torch.Tensor], save_path: Optional[str] = None):
    """적응 과정에서 z 벡터의 변화 시각화"""
    if len(z_history) == 0:
        return
    
    z_array = np.array([z.cpu().numpy().flatten() for z in z_history])
    n_steps, z_dim = z_array.shape
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 1. 각 차원별 변화
    ax1 = axes[0]
    n_dims_to_show = min(8, z_dim)
    for i in range(n_dims_to_show):
        ax1.plot(range(n_steps), z_array[:, i], label=f'z[{i}]', alpha=0.7, marker='o')
    ax1.set_xlabel('Adaptation Step')
    ax1.set_ylabel('z value')
    ax1.set_title('Latent Vector z Evolution (First 8 Dimensions)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. L2 norm
    ax2 = axes[1]
    z_norms = np.linalg.norm(z_array, axis=1)
    ax2.plot(range(n_steps), z_norms, 'r-', linewidth=2, marker='s')
    ax2.set_xlabel('Adaptation Step')
    ax2.set_ylabel('||z||_2')
    ax2.set_title('Latent Vector Magnitude')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_adaptation_summary(z_history: List[torch.Tensor], context_sizes: List[int],
                           save_path: Optional[str] = None):
    """적응 과정 전체 요약"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    z_array = np.array([z.cpu().numpy().flatten() for z in z_history])
    n_steps = len(z_history)
    
    # 1. PCA 투영
    ax1 = axes[0, 0]
    if z_array.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(z_array)
        ax1.plot(z_pca[:, 0], z_pca[:, 1], 'o-', linewidth=2, markersize=8)
        ax1.scatter(z_pca[0, 0], z_pca[0, 1], c='green', s=200, marker='*', label='Initial', zorder=5)
        ax1.scatter(z_pca[-1, 0], z_pca[-1, 1], c='red', s=200, marker='*', label='Final', zorder=5)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    else:
        ax1.plot(z_array[:, 0], z_array[:, 1], 'o-', linewidth=2, markersize=8)
        ax1.scatter(z_array[0, 0], z_array[0, 1], c='green', s=200, marker='*', label='Initial', zorder=5)
        ax1.scatter(z_array[-1, 0], z_array[-1, 1], c='red', s=200, marker='*', label='Final', zorder=5)
        ax1.set_xlabel('z[0]')
        ax1.set_ylabel('z[1]')
    ax1.set_title('Latent Space Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Context 크기
    ax2 = axes[0, 1]
    ax2.plot(range(n_steps), context_sizes, 's-', linewidth=2, markersize=10)
    ax2.set_xlabel('Adaptation Step')
    ax2.set_ylabel('Context Size (K)')
    ax2.set_title('Context Growth')
    ax2.grid(True, alpha=0.3)
    
    # 3. L2 norm
    ax3 = axes[1, 0]
    z_norms = np.linalg.norm(z_array, axis=1)
    ax3.plot(range(n_steps), z_norms, 'r-', linewidth=2, marker='o')
    ax3.set_xlabel('Adaptation Step')
    ax3.set_ylabel('||z||_2')
    ax3.set_title('Latent Vector Magnitude')
    ax3.grid(True, alpha=0.3)
    
    # 4. 변화율
    ax4 = axes[1, 1]
    if n_steps > 1:
        z_diffs = np.linalg.norm(np.diff(z_array, axis=0), axis=1)
        ax4.plot(range(1, n_steps), z_diffs, 'g-', linewidth=2, marker='s')
        ax4.set_xlabel('Adaptation Step')
        ax4.set_ylabel('||Δz||_2')
        ax4.set_title('Step-wise Change Magnitude')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Need at least 2 steps', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Step-wise Change Magnitude')
    
    plt.suptitle('Adaptation Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 5. 최종 성과 비교 시각화
# ============================================================================

def plot_before_after_comparison(trajectories: List[Dict], reward_before: Callable,
                                 reward_after: Callable, save_path: Optional[str] = None):
    """적응 전/후 보상 비교"""
    scores_before, scores_after = [], []
    
    for traj in trajectories:
        total_before = sum(reward_before(s, a) for s, a in zip(traj['observations'], traj['actions']))
        total_after = sum(reward_after(s, a) for s, a in zip(traj['observations'], traj['actions']))
        avg_before = total_before / len(traj['observations'])
        avg_after = total_after / len(traj['observations'])
        scores_before.append(avg_before)
        scores_after.append(avg_after)
    
    scores_before = np.array(scores_before)
    scores_after = np.array(scores_after)
    reward_change = scores_after - scores_before
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 분포 비교
    axes[0, 0].hist(scores_before, bins=30, alpha=0.6, label='Before', color='blue', edgecolor='black')
    axes[0, 0].hist(scores_after, bins=30, alpha=0.6, label='After', color='red', edgecolor='black')
    axes[0, 0].axvline(scores_before.mean(), color='blue', linestyle='--', linewidth=2, alpha=0.8)
    axes[0, 0].axvline(scores_after.mean(), color='red', linestyle='--', linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('Average Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Reward Distribution: Before vs After')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 산점도
    axes[0, 1].scatter(scores_before, scores_after, alpha=0.5, s=20)
    min_val = min(scores_before.min(), scores_after.min())
    max_val = max(scores_before.max(), scores_after.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='No Change')
    axes[0, 1].set_xlabel('Reward (Before)')
    axes[0, 1].set_ylabel('Reward (After)')
    axes[0, 1].set_title('Reward Change: Before → After')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 변화량 분포
    axes[1, 0].hist(reward_change, bins=30, alpha=0.7, edgecolor='black', color='green')
    axes[1, 0].axvline(0, color='r', linestyle='--', linewidth=2, label='No Change')
    axes[1, 0].axvline(reward_change.mean(), color='b', linestyle='--', 
                      linewidth=2, label=f'Mean: {reward_change.mean():.4f}')
    axes[1, 0].set_xlabel('Reward Change (After - Before)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Reward Change Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 통계 요약
    axes[1, 1].axis('off')
    stats_text = f"""
Statistics Summary:

Before Adaptation:
  Mean: {scores_before.mean():.4f}
  Std:  {scores_before.std():.4f}
  Min:  {scores_before.min():.4f}
  Max:  {scores_before.max():.4f}

After Adaptation:
  Mean: {scores_after.mean():.4f}
  Std:  {scores_after.std():.4f}
  Min:  {scores_after.min():.4f}
  Max:  {scores_after.max():.4f}

Change:
  Mean Change: {reward_change.mean():.4f}
  Std Change:  {reward_change.std():.4f}
  Improved:    {(reward_change > 0).sum()} / {len(reward_change)} ({100*(reward_change > 0).mean():.1f}%)
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Adaptation Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reward_distribution(trajectories: List[Dict], reward_function: Callable,
                            top_k: int = 10, save_path: Optional[str] = None):
    """보상 분포 시각화"""
    trajectory_scores = []
    for traj in trajectories:
        total_reward = sum(reward_function(s, a) for s, a in zip(traj['observations'], traj['actions']))
        avg_reward = total_reward / len(traj['observations'])
        trajectory_scores.append(avg_reward)
    
    trajectory_scores = np.array(trajectory_scores)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 분포
    axes[0].hist(trajectory_scores, bins=30, alpha=0.7, edgecolor='black')
    axes[0].axvline(trajectory_scores.mean(), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {trajectory_scores.mean():.4f}')
    axes[0].set_xlabel('Average Reward')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Reward Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 상위 k개
    axes[1].barh(range(top_k), np.sort(trajectory_scores)[-top_k:][::-1], alpha=0.7)
    axes[1].set_yticks(range(top_k))
    axes[1].set_yticklabels([f'#{i+1}' for i in range(top_k)])
    axes[1].set_xlabel('Average Reward')
    axes[1].set_title(f'Top {top_k} Trajectories')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 6. 전체 파이프라인 요약 시각화
# ============================================================================

def create_pipeline_summary(data_stats: Dict, training_stats: Dict, 
                           adaptation_stats: Dict, save_path: Optional[str] = None):
    """전체 파이프라인 성과 요약"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 제목
    fig.suptitle('Project Pipeline Summary', fontsize=18, fontweight='bold', y=0.98)
    
    # 1. 데이터 생성 통계
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    data_text = f"""
Data Generation:
  Trajectories: {data_stats.get('num_trajectories', 'N/A')}
  Features: {data_stats.get('num_features', 'N/A')}
  Avg Jerk: {data_stats.get('avg_jerk', 'N/A'):.4f}
  Avg Pitch: {data_stats.get('avg_pitch', 'N/A'):.4f}
    """
    ax1.text(0.1, 0.5, data_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 2. 선호도 데이터셋 통계
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    pref_text = f"""
Preference Dataset:
  Pairs: {data_stats.get('num_pairs', 'N/A')}
  User Groups: {data_stats.get('num_groups', 'N/A')}
  Avg Pairs/Group: {data_stats.get('avg_pairs_per_group', 'N/A'):.1f}
    """
    ax2.text(0.1, 0.5, pref_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 3. 학습 통계
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    train_text = f"""
Training:
  Epochs: {training_stats.get('num_epochs', 'N/A')}
  Final Loss: {training_stats.get('final_loss', 'N/A'):.4f}
  Final Accuracy: {training_stats.get('final_accuracy', 'N/A'):.4f}
  Best Epoch: {training_stats.get('best_epoch', 'N/A')}
    """
    ax3.text(0.1, 0.5, train_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # 4. 적응 통계
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    adapt_text = f"""
Adaptation:
  Steps: {adaptation_stats.get('num_steps', 'N/A')}
  Final Context Size: {adaptation_stats.get('final_context_size', 'N/A')}
  z Change: {adaptation_stats.get('z_change', 'N/A'):.4f}
    """
    ax4.text(0.1, 0.5, adapt_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # 5-9. 차트들 (플레이스홀더)
    for i in range(1, 3):
        for j in range(1, 3):
            ax = fig.add_subplot(gs[i, j])
            ax.text(0.5, 0.5, f'Chart {i*3+j}', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

