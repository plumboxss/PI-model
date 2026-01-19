import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PreprocessStats:
    """
    Training-time preprocessing statistics to ensure train/adapt/eval consistency.

    - obs_dim_used: number of observation dims used (x_com removed => 9)
    - downsample_step: temporal downsample step (T -> T/downsample_step)
    - obs_mean/std, act_mean/std: Z-score stats computed on the TRAIN preference dataset
    """

    obs_dim_used: int
    downsample_step: int
    obs_mean: np.ndarray
    obs_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray

    @staticmethod
    def from_npz(path: str) -> "PreprocessStats":
        data = np.load(path, allow_pickle=False)
        return PreprocessStats(
            obs_dim_used=int(data["obs_dim_used"]),
            downsample_step=int(data["downsample_step"]),
            obs_mean=data["obs_mean"].astype(np.float32),
            obs_std=data["obs_std"].astype(np.float32),
            act_mean=data["act_mean"].astype(np.float32),
            act_std=data["act_std"].astype(np.float32),
        )

    def to_npz(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.savez(
            path,
            obs_dim_used=np.array(self.obs_dim_used, dtype=np.int64),
            downsample_step=np.array(self.downsample_step, dtype=np.int64),
            obs_mean=self.obs_mean.astype(np.float32),
            obs_std=self.obs_std.astype(np.float32),
            act_mean=self.act_mean.astype(np.float32),
            act_std=self.act_std.astype(np.float32),
        )


def infer_preprocess_stats_path(vae_model_path: str) -> str:
    """Default location: same directory as the saved model checkpoint."""
    return os.path.join(os.path.dirname(vae_model_path), "preprocessing_stats.npz")


def preprocess_trajectory(
    obs: np.ndarray,
    act: np.ndarray,
    stats: PreprocessStats,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the exact preprocessing used in training:
    - drop x_com by taking first obs_dim_used dims
    - temporal downsample
    - Z-score normalize using training stats
    """
    if obs.ndim != 2:
        raise ValueError(f"obs must be 2D (T, obs_dim), got shape={obs.shape}")
    if act.ndim != 2:
        raise ValueError(f"act must be 2D (T, act_dim), got shape={act.shape}")

    obs = obs[:, : stats.obs_dim_used]

    ds = slice(None, None, stats.downsample_step)
    obs = obs[ds]
    act = act[ds]

    obs = (obs - stats.obs_mean) / (stats.obs_std + 1e-8)
    act = (act - stats.act_mean) / (stats.act_std + 1e-8)

    return obs.astype(np.float32), act.astype(np.float32)


def preprocess_state_action(
    s: np.ndarray,
    a: np.ndarray,
    stats: PreprocessStats,
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess a single timestep (no downsampling applied here)."""
    s = s[: stats.obs_dim_used]
    s = (s - stats.obs_mean) / (stats.obs_std + 1e-8)
    a = (a - stats.act_mean) / (stats.act_std + 1e-8)
    return s.astype(np.float32), a.astype(np.float32)


