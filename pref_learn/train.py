import os
from collections import defaultdict

import absl.app
import absl.flags
import gym
import numpy as np
import torch

import jaxrl_m.envs
from pref_learn.utils.utils import (
    define_flags_with_default,
    set_random_seed,
    get_user_flags,
    WandBLogger,
    prefix_metrics,
)
from pref_learn.models.utils import get_datasets, Annealer, EarlyStopper
from pref_learn.models.vae import VAEModel
import pref_learn.utils.plot_utils as putils

FLAGS_DEF = define_flags_with_default(
    env="maze2d-target-v0",  # can change
    comment="",
    data_seed=42,
    batch_size=256,
    set_size=-1,
    early_stop=False,
    min_delta=3e-4,
    patience=10,
    lr=1e-3,
    model_type="VAE",  # VAE로 고정
    # Attention Encoder
    encoder_type='attention', # 'mlp' or 'attention'
    hidden_dim=256,
    n_heads=4,
    n_layers=2,
    # VAE
    latent_dim=32,
    kl_weight=1.0,
    learned_prior=False,
    use_annealing=False,
    annealer_baseline=0.0,
    annealer_type="cosine",
    annealer_cycles=4,
    # Training
    n_epochs=500,
    eval_freq=50,
    save_freq=50,
    device="cuda",
    # Dataset
    dataset_path="",
    logging=WandBLogger.get_default_config(),
    seed=42,
    # plotting
    debug_plots=False,
    plot_observations=False,
    reward_scaling=1.0,
    # biased
    biased_mode="grid",
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

    gym_env = gym.make(FLAGS.env)
    gym_env.seed(FLAGS.seed)
    gym_env.action_space.seed(FLAGS.seed)
    gym_env.observation_space.seed(FLAGS.seed)
    set_random_seed(FLAGS.seed)
    if hasattr(gym_env, "reward_observation_space"):
        observation_dim = gym_env.reward_observation_space.shape[0]
    else:
        observation_dim = gym_env.observation_space.shape[0]
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
        FLAGS.encoder_type
    )

    annealer = None
    if FLAGS.use_annealing:
        annealer = Annealer(
            total_steps=FLAGS.n_epochs // FLAGS.annealer_cycles,
            shape=FLAGS.annealer_type,
            baseline=FLAGS.annealer_baseline,
            cyclical=FLAGS.annealer_cycles > 1,
        )
    
    decoder_input_dim = observation_dim + action_dim + FLAGS.latent_dim

    reward_model = VAEModel(
        encoder_input_dim=encoder_input_dim,
        decoder_input_dim=decoder_input_dim,
        action_dim=action_dim,
        latent_dim=FLAGS.latent_dim,
        hidden_dim=FLAGS.hidden_dim,
        kl_weight=FLAGS.kl_weight,
        learned_prior=FLAGS.learned_prior,
        annealer=annealer,
        reward_scaling=FLAGS.reward_scaling,
        encoder_type=FLAGS.encoder_type,
        n_heads=FLAGS.n_heads,
        n_layers=FLAGS.n_layers,
    )

    device = FLAGS.device
    reward_model = reward_model.to(device)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=FLAGS.lr)
    early_stop = EarlyStopper(FLAGS.patience, FLAGS.min_delta)
    best_criteria = None
    for epoch in range(FLAGS.n_epochs):
        metrics = defaultdict(list)
        metrics["epoch"] = epoch

        for batch in train_loader:
            s1 = batch["s1"].to(device).float()
            s2 = batch["s2"].to(device).float()
            labels = batch["labels"].to(device).float()

            if FLAGS.encoder_type == 'attention':
                # For attention, the batch from GroupBatchSampler is (Batch=Context, 1, T, D)
                # We need to reshape it to (1, Context, T, D) to treat the batch as one context
                s1 = s1.unsqueeze(0)
                s2 = s2.unsqueeze(0)
                labels = labels.unsqueeze(0)
            
            optimizer.zero_grad()
            # 모델은 이제 s1, s2, labels만 받습니다. 내부에서 obs/act를 분리합니다.
            loss, batch_metrics = reward_model(s1, s2, labels)
            loss.backward()
            optimizer.step()

            for key, val in prefix_metrics(batch_metrics, "train").items():
                metrics[key].append(val)

        if epoch % FLAGS.eval_freq == 0:
            for batch in test_loader:
                with torch.no_grad():
                    s1 = batch["s1"].to(device).float()
                    s2 = batch["s2"].to(device).float()
                    labels = batch["labels"].to(device).float()

                    if FLAGS.encoder_type == 'attention':
                        # Same reshaping for evaluation
                        s1 = s1.unsqueeze(0)
                        s2 = s2.unsqueeze(0)
                        labels = labels.unsqueeze(0)

                    loss, batch_metrics = reward_model(
                        s1, s2, labels
                    )

                    for key, val in prefix_metrics(batch_metrics, "eval").items():
                        metrics[key].append(val)

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


if __name__ == "__main__":
    absl.app.run(main)
