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
    jerks, pitches, settling_times = [], [], []
    
    for traj in trajectories.values():
        if 'features' in traj and traj['features']:
            features = traj['features']
            if 'jerk' in features:
                jerks.append(features['jerk'])
            if 'pitch' in features:
                pitches.append(features['pitch'])
            if 'settling_time' in features:
                settling_times.append(features['settling_time'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
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
    ax2.bar(range(len(unique_model_ids)), model_counts, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Model ID (User Group)')
    ax2.set_ylabel('Number of Pairs')
    ax2.set_title('Preference Pairs per User Group')
    ax2.set_xticks(range(len(unique_model_ids)))
    ax2.set_xticklabels([f'G{i}' for i in unique_model_ids], rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
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
    
    epochs = range(len(metrics_history.get('train/loss', [])))
    
    # 1. Loss
    ax1 = axes[0, 0]
    if 'train/loss' in metrics_history:
        ax1.plot(epochs, metrics_history['train/loss'], label='Train Loss', alpha=0.7)
    if 'eval/loss' in metrics_history:
        ax1.plot(epochs, metrics_history['eval/loss'], label='Eval Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy
    ax2 = axes[0, 1]
    if 'train/accuracy' in metrics_history:
        ax2.plot(epochs, metrics_history['train/accuracy'], label='Train Accuracy', alpha=0.7)
    if 'eval/accuracy' in metrics_history:
        ax2.plot(epochs, metrics_history['eval/accuracy'], label='Eval Accuracy', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. KL Divergence
    ax3 = axes[1, 0]
    if 'train/kld_loss' in metrics_history:
        ax3.plot(epochs, metrics_history['train/kld_loss'], label='Train KL', alpha=0.7)
    if 'eval/kld_loss' in metrics_history:
        ax3.plot(epochs, metrics_history['eval/kld_loss'], label='Eval KL', alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('KL Divergence')
    ax3.set_title('KL Divergence Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. KL Weight (Annealing)
    ax4 = axes[1, 1]
    if 'train/kl_weight' in metrics_history:
        ax4.plot(epochs, metrics_history['train/kl_weight'], label='KL Weight', alpha=0.7, color='green')
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

