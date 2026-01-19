import os
import sys
from collections import defaultdict

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import absl.app
import absl.flags
import gym
import numpy as np
import torch
import wandb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from src.models.vae import VAEModel
from src.data.loader import get_datasets
from src.utils.training_utils import define_flags_with_default, set_seed, WandBLogger, get_user_flags
from src.utils.plot_utils import AnnealedLinearSchedule
from src.utils.training_utils import EarlyStopper, prefix_metrics, Annealer
import src.utils.plot_utils as putils
from src.simulation.env import SingleScenarioEnv
from src.utils.preprocessing import PreprocessStats

FLAGS_DEF = define_flags_with_default(
    seed=42,
    # env_name='HalfCheetah-v2',
    max_steps=int(2e5),
    batch_size=256,
    log_interval=1000,
    set_size=-1,
    early_stop=False,
    min_delta=3e-4,
    patience=10,
    lr=1e-3,
    model_type="VAE",  # VAE로 고정
    # Encoder types
    trajectory_encoder_type='mlp',  # 'transformer', 'lstm', 'mlp'
    set_encoder_type='attention',  # 'attention', 'deepset'
    hidden_dim=256,
    n_heads=4,
    n_layers=2,
    # VAE
    latent_dim=4,
    kl_weight=1.0,
    kl_max=0.5,  # Cap for annealed KL weight to mitigate posterior collapse
    learned_prior=False,
    use_annealing=True, # 기본값 변경: False -> True
    annealer_baseline=0.0,
    annealer_type="cosine",
    annealer_cycles=4,
    # Training
    n_epochs=500,
    eval_freq=50,
    save_freq=50,
    device="cuda",
    env='Suspension-v0', # Default env name to avoid flag error
    # Dataset
    dataset_path="",
    context_size=15,  # Number of context comparisons (K)
    logging=WandBLogger.get_default_config(),
    # seed=42, # Duplicate seed definition removed
    # plotting
    debug_plots=False,
    plot_observations=False,
    reward_scaling=1000.0,
    free_bits=0.15,  # KL floor (nats per latent dimension)
    decoder_feature_dropout=0.1,  # Weak dropout in RewardDecoder.feature_net
    # biased
    biased_mode="grid",
    comment="", # Add comment flag
    save_training_curves=True,  # Save training curves to local file
)


def log_metrics(metrics, epoch, logger):
    for key, val in metrics.items():
        if isinstance(val, list):
            metrics[key] = np.mean(val)
    logger.log(metrics, step=epoch)


def main(_):
    FLAGS = absl.flags.FLAGS
    assert os.path.exists(FLAGS.dataset_path), "You must provide a dataset path."
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    save_dir = FLAGS.logging.output_dir + "/" + FLAGS.env
    save_dir += "/" + str(FLAGS.model_type) + "/"

    FLAGS.logging.group = f"{FLAGS.env}_{FLAGS.model_type}"
    assert FLAGS.comment, "You must leave your comment for logging experiment."
    FLAGS.logging.group += f"_{FLAGS.comment}"
    FLAGS.logging.experiment_id = FLAGS.logging.group + f"_s{FLAGS.seed}"
    save_dir += f"{FLAGS.comment}" + "/"
    save_dir += "s" + str(FLAGS.seed)
    FLAGS.logging.output_dir = save_dir
    wb_logger = WandBLogger(FLAGS.logging, variant=variant)

    gym_env = SingleScenarioEnv()
    # gym_env.seed(FLAGS.seed) # Removed deprecated seed method
    gym_env.action_space.seed(FLAGS.seed)
    gym_env.observation_space.seed(FLAGS.seed)
    # set_random_seed(FLAGS.seed) # Moved to training_utils
    set_seed(FLAGS.seed)
    # Loader에서 x_com(마지막 1차원)을 제거하고 정규화하므로 여기서도 맞춰서 -1 처리
    if hasattr(gym_env, "reward_observation_space"):
        observation_dim = gym_env.reward_observation_space.shape[0]
    else:
        observation_dim = gym_env.observation_space.shape[0]
    observation_dim = observation_dim - 1  # drop x_com
    assert observation_dim > 0, "observation_dim must be positive after removing x_com"
    if "maze" in FLAGS.env:
        gym_env.set_biased_mode(FLAGS.biased_mode)
    action_dim = gym_env.action_space.shape[0]

    (
        train_loader,
        test_loader,
        train_dataset,
        eval_dataset,
        len_set,
        len_query,
        encoder_input_dim,
    ) = get_datasets(
        FLAGS.dataset_path,
        observation_dim,
        action_dim,
        FLAGS.batch_size,
        FLAGS.set_size,
        FLAGS.set_encoder_type,
        FLAGS.context_size,
        split_seed=FLAGS.seed  # Use same seed for train/test split reproducibility
    )

    # Save training-time preprocessing stats for strict train/adapt/eval consistency (P0)
    # This is critical for few-shot adaptation: the encoder/decoder were trained on normalized & downsampled inputs.
    try:
        stats = PreprocessStats(
            obs_dim_used=int(getattr(train_dataset, "obs_mean").shape[0]),
            downsample_step=int(getattr(train_dataset, "downsample_step")),
            obs_mean=np.array(getattr(train_dataset, "obs_mean")),
            obs_std=np.array(getattr(train_dataset, "obs_std")),
            act_mean=np.array(getattr(train_dataset, "act_mean")),
            act_std=np.array(getattr(train_dataset, "act_std")),
        )
        stats_path = os.path.join(save_dir, "preprocessing_stats.npz")
        stats.to_npz(stats_path)
        print(f"✅ Saved preprocessing stats to: {stats_path}")
    except Exception as e:
        print(f"⚠️  Warning: Failed to save preprocessing stats (adapt/eval may mismatch training): {e}")

    annealer = None
    if FLAGS.use_annealing:
        annealer = Annealer(
            total_steps=FLAGS.n_epochs // FLAGS.annealer_cycles,
            shape=FLAGS.annealer_type,
            baseline=FLAGS.annealer_baseline,
            cyclical=FLAGS.annealer_cycles > 1,
        )
    
    reward_model = VAEModel(
        obs_dim=observation_dim,
        act_dim=action_dim,
        latent_dim=FLAGS.latent_dim,
        hidden_dim=FLAGS.hidden_dim,
        kl_weight=FLAGS.kl_weight,
        kl_max=FLAGS.kl_max,
        learned_prior=FLAGS.learned_prior,
        annealer=annealer,
        reward_scaling=FLAGS.reward_scaling,
        decoder_feature_dropout=FLAGS.decoder_feature_dropout,
        trajectory_encoder_type=FLAGS.trajectory_encoder_type,
        set_encoder_type=FLAGS.set_encoder_type,
        n_heads=FLAGS.n_heads,
        n_layers=FLAGS.n_layers,
        free_bits=FLAGS.free_bits,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FLAGS.device = str(device)
    reward_model = reward_model.to(device)
    # Use a lower LR for decoder to prevent it from overpowering the encoder
    decoder_lr = FLAGS.lr * 0.2  # 1/5 of encoder LR
    decoder_params = list(reward_model.reward_decoder.parameters())
    encoder_params = [
        p for name, p in reward_model.named_parameters()
        if not name.startswith("reward_decoder.")
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": encoder_params, "lr": FLAGS.lr},
            {"params": decoder_params, "lr": decoder_lr},
        ]
    )
    early_stop = EarlyStopper(FLAGS.patience, FLAGS.min_delta)
    best_criteria = None
    # 전체 학습 히스토리 저장 (시각화용)
    metrics_history = defaultdict(list)
    
    for epoch in range(FLAGS.n_epochs):
        metrics = defaultdict(list)
        metrics["epoch"] = epoch

        for batch_idx, batch in enumerate(train_loader):
            # Extract context and query data
            context_s1 = batch["context_s1"].to(device).float()
            context_s2 = batch["context_s2"].to(device).float()
            context_y = batch["context_y"].to(device).float()
            query_s1 = batch["query_s1"].to(device).float()
            query_s2 = batch["query_s2"].to(device).float()
            query_y = batch["query_y"].to(device).float()
            
            # Shape validation (first batch only)
            if batch_idx == 0 and epoch == 0:
                print(f"\n=== Shape Validation ===")
                print(f"context_s1: {context_s1.shape} (B, K, T, D_sa)")
                print(f"context_s2: {context_s2.shape} (B, K, T, D_sa)")
                print(f"context_y: {context_y.shape} (B, K, 1)")
                print(f"query_s1: {query_s1.shape} (B, T, D_sa)")
                print(f"query_s2: {query_s2.shape} (B, T, D_sa)")
                print(f"query_y: {query_y.shape} (B, 1)")
                print(f"=======================\n")
                
                # Assertions
                B, K, T, D_sa = context_s1.shape
                assert context_s2.shape == (B, K, T, D_sa), f"context_s2 shape mismatch: {context_s2.shape}"
                assert context_y.shape == (B, K, 1), f"context_y shape mismatch: {context_y.shape}"
                assert query_s1.shape == (B, T, D_sa), f"query_s1 shape mismatch: {query_s1.shape}"
                assert query_s2.shape == (B, T, D_sa), f"query_s2 shape mismatch: {query_s2.shape}"
                assert query_y.shape == (B, 1), f"query_y shape mismatch: {query_y.shape}"
                assert D_sa == observation_dim + action_dim, f"D_sa mismatch: {D_sa} != {observation_dim + action_dim}"
            
            optimizer.zero_grad()
            # Forward pass with context-query structure
            loss, batch_metrics = reward_model(
                context_s1, context_s2, context_y,
                query_s1, query_s2, query_y
            )
            # Remove p_hat from logging metrics if present
            if "p_hat" in batch_metrics:
                batch_metrics.pop("p_hat")
            loss.backward()
            optimizer.step()
            
            if batch_idx % 1 == 0: # Print every batch to verify speed
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

            for key, val in prefix_metrics(batch_metrics, "train").items():
                metrics[key].append(val)

        if epoch % FLAGS.eval_freq == 0:
            roc_y_true = []
            roc_y_score = []
            for batch in test_loader:
                with torch.no_grad():
                    # Extract context and query data
                    context_s1 = batch["context_s1"].to(device).float()
                    context_s2 = batch["context_s2"].to(device).float()
                    context_y = batch["context_y"].to(device).float()
                    query_s1 = batch["query_s1"].to(device).float()
                    query_s2 = batch["query_s2"].to(device).float()
                    query_y = batch["query_y"].to(device).float()

                    loss, batch_metrics = reward_model(
                        context_s1, context_s2, context_y,
                        query_s1, query_s2, query_y
                    )

                    p_hat = batch_metrics.pop("p_hat", None)
                    for key, val in prefix_metrics(batch_metrics, "eval").items():
                        metrics[key].append(val)

                    if p_hat is not None:
                        roc_y_score.append(p_hat.detach().cpu().numpy())
                        roc_y_true.append(query_y.detach().cpu().numpy())

            if FLAGS.debug_plots and "maze2d" in FLAGS.env:
                fig_dict = putils.plot_vae(
                    gym_env,
                    reward_model,
                    eval_dataset,
                    classifier=False, # VAEClassifier는 제거되었으므로 False
                )
                metrics.update(prefix_metrics(fig_dict, "debug_plots"))
            else:
                putils.update_posterior(gym_env, reward_model, eval_dataset)

            if len(roc_y_true) > 0 and len(roc_y_score) > 0:
                y_true = np.concatenate(roc_y_true, axis=0).reshape(-1)
                y_score = np.concatenate(roc_y_score, axis=0).reshape(-1)
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                metrics["eval/roc_auc"] = roc_auc

                roc_path = os.path.join(save_dir, f"roc_epoch_{epoch}.png")
                plt.figure()
                plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(roc_path)
                plt.close()
            

            criteria = np.mean(metrics["eval/loss"])

            if best_criteria is None:
                best_criteria = criteria
                torch.save(reward_model, save_dir + f"/best_model.pt")

            if criteria < best_criteria:
                torch.save(reward_model, save_dir + f"/best_model.pt")
                best_criteria = criteria

            if FLAGS.early_stop and early_stop.early_stop(criteria):
                log_metrics(metrics, epoch, wb_logger)
                torch.save(reward_model, save_dir + f"/model_{epoch}.pt")
                break

        if epoch % FLAGS.save_freq == 0:
            torch.save(reward_model, save_dir + f"/model_{epoch}.pt")

        if FLAGS.use_annealing:
            reward_model.annealer.step()

        log_metrics(metrics, epoch, wb_logger)
        
        # 메트릭 히스토리 업데이트 (시각화용)
        for key, val in metrics.items():
            if isinstance(val, list):
                avg_val = np.mean(val)
            else:
                avg_val = val
            metrics_history[key].append(avg_val)
        
        # 학습 곡선 로컬 저장 (마지막 epoch 또는 주기적으로)
        if FLAGS.save_training_curves and (epoch == FLAGS.n_epochs - 1 or epoch % FLAGS.save_freq == 0):
            from src.utils.visualization import plot_training_curves
            
            # 메트릭 히스토리 준비 (eval 메트릭은 eval_freq에 맞춰 필터링)
            plot_metrics = {}
            for key in ['train/loss', 'eval/loss', 'train/accuracy', 'eval/accuracy', 
                       'train/kld_loss', 'eval/kld_loss',
                       'train/kld_loss_raw', 'eval/kld_loss_raw',
                       'train/kl_weight']:
                if key in metrics_history:
                    plot_metrics[key] = metrics_history[key]
            
            plot_training_curves(
                plot_metrics,
                save_path=os.path.join(save_dir, 'training_curves.png')
            )


if __name__ == "__main__":
    absl.app.run(main)
