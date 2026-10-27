"""Checkpoint loading, batched inference and kinematic skill metrics."""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import NormStats, create_stacked_train_val_datasets
from .kinematics import DX_M, F0, divergence, strain, vorticity
from .models import UNet

DEEP_CHANNELS = ["Eta", "windx", "windy", "theta", "u1", "v1"]
DEEP_GRAD_CHANNELS = ["a", "b", "windx", "windy", "theta", "u1", "v1"]
OUTPUT_VARS = ["u1", "v1"]


@dataclass(frozen=True)
class PaperModel:
    key: str            # file-name friendly id
    label: str          # short label used in the paper
    exp_dir: str        # relative to the results root
    data_file: str
    channel_order: Tuple[str, ...]
    input_vars: Tuple[str, ...]
    base_channels: int


PAPER_MODELS: List[PaperModel] = [
    PaperModel("ssh_all", "SSH+All", "ssha_followup/ssha_sst_wind_double_model",
               "global-final-64_stacked_DEEP.npy", tuple(DEEP_CHANNELS),
               ("Eta", "windx", "windy", "theta"), 192),
    PaperModel("grad_all", "Grad+All", "grad_bc192/grad_ab_sst_wind_bc192",
               "global-final-64_stacked_DEEP_GRAD.npy", tuple(DEEP_GRAD_CHANNELS),
               ("a", "b", "theta", "windx", "windy"), 192),
    PaperModel("grad_all_lg", "Grad+All-Lg", "grad_bc384/grad_ab_sst_wind_bc384",
               "global-final-64_stacked_DEEP_GRAD.npy", tuple(DEEP_GRAD_CHANNELS),
               ("a", "b", "theta", "windx", "windy"), 384),
    PaperModel("grad_all_lg_mse", "Grad+All-Lg/MSE", "grad_bc384/grad_ab_sst_wind_bc384_mse",
               "global-final-64_stacked_DEEP_GRAD.npy", tuple(DEEP_GRAD_CHANNELS),
               ("a", "b", "theta", "windx", "windy"), 384),
    PaperModel("grad_all_lg_phys", "Grad+All-Lg/Phys", "grad_bc384/grad_ab_sst_wind_bc384_l1vd",
               "global-final-64_stacked_DEEP_GRAD.npy", tuple(DEEP_GRAD_CHANNELS),
               ("a", "b", "theta", "windx", "windy"), 384),
]
PAPER_MODELS_BY_KEY = {m.key: m for m in PAPER_MODELS}


def load_checkpoint(path: str, in_channels: int, base_channels: int,
                    device: torch.device) -> Tuple[torch.nn.Module, NormStats]:
    """Load a training checkpoint; returns (model in eval mode, normalisation stats)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model = UNet(in_channels=in_channels, out_channels=2, base=base_channels)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    ns = ckpt["norm_stats"]
    stats = NormStats(
        x_mean=np.asarray(ns["x_mean"], dtype=np.float32),
        x_std=np.asarray(ns["x_std"], dtype=np.float32),
        y_mean=np.asarray(ns["y_mean"], dtype=np.float32),
        y_std=np.asarray(ns["y_std"], dtype=np.float32),
    )
    return model, stats


def validation_dataset(data_dir: str, data_file: str, channel_order, input_vars,
                       stats: Optional[NormStats] = None, seed: int = 42,
                       val_fraction: float = 0.2):
    """Re-create the validation split used in training.

    The released runs fell back to a random 80/20 split of patches with seed 42
    because no temporal-split metadata was present; this reproduces it exactly.
    """
    # normalize=False skips the (slow) re-estimation of normalisation statistics;
    # the checkpoint's own statistics are attached afterwards.
    _, val = create_stacked_train_val_datasets(
        data_dir=data_dir, data_file=data_file, channel_order=list(channel_order),
        input_vars=list(input_vars), output_vars=OUTPUT_VARS, val_fraction=val_fraction,
        normalize=False, seed=seed, use_precomputed_split=True, verbose=False,
    )
    if stats is not None:
        val.stats = stats
        val.normalize = True
    return val


@torch.no_grad()
def predict(model, dataset, stats: NormStats, device: torch.device,
            batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred, truth, mask) in physical units; shapes (N, 2, H, W), (N, 2, H, W), (N, H, W)."""
    y_mean = torch.from_numpy(stats.y_mean).float().view(1, 2, 1, 1).to(device)
    y_std = torch.from_numpy(stats.y_std).float().view(1, 2, 1, 1).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    preds, truths, masks = [], [], []
    for x, y, m in tqdm(loader, desc="inference", leave=False):
        x, y = x.to(device), y.to(device)
        preds.append((model(x).float() * y_std + y_mean).cpu().numpy())
        truths.append((y.float() * y_std + y_mean).cpu().numpy())
        masks.append(m[:, 0].numpy() > 0.5)
    return np.concatenate(preds), np.concatenate(truths), np.concatenate(masks)


def velocity_metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray,
                     speed_eps: float = 0.01) -> Dict[str, float]:
    """Pointwise velocity errors over valid pixels (same definitions as training)."""
    up, vp = pred[:, 0][mask], pred[:, 1][mask]
    ut, vt = truth[:, 0][mask], truth[:, 1][mask]
    sp, st = np.hypot(up, vp), np.hypot(ut, vt)
    cos = np.clip((up * ut + vp * vt) / (sp * st + 1e-8), -1.0, 1.0)
    keep = st > speed_eps
    return {
        "rmse_u": float(np.sqrt(np.mean((up - ut) ** 2))),
        "rmse_v": float(np.sqrt(np.mean((vp - vt) ** 2))),
        "rmse_speed": float(np.sqrt(np.mean((sp - st) ** 2))),
        "mean_dir_error_deg": float(np.degrees(np.arccos(cos[keep])).mean()),
    }


def _kinematic_fields(uv: np.ndarray, dx: float, chunk: int = 1024):
    """Vorticity, divergence and strain for (N, 2, H, W) velocities, computed in chunks."""
    z, d, s = [], [], []
    for i in range(0, len(uv), chunk):
        t = torch.from_numpy(np.ascontiguousarray(uv[i:i + chunk])).double()
        u, v = t[:, 0], t[:, 1]
        z.append(vorticity(u, v, dx).float().numpy())
        d.append(divergence(u, v, dx).float().numpy())
        s.append(strain(u, v, dx).float().numpy())
    return np.concatenate(z), np.concatenate(d), np.concatenate(s)


def kinematic_metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray,
                      dx: float = DX_M, f: float = F0) -> Dict[str, float]:
    """RMSE and Pearson r of zeta/f and div/f (signed f; dimensionless) over valid pixels."""
    zp, dp, _ = _kinematic_fields(pred, dx)
    zt, dt, _ = _kinematic_fields(truth, dx)
    zp, zt = zp[mask] / f, zt[mask] / f
    dp, dt = dp[mask] / f, dt[mask] / f

    def r(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")

    rmse_z = float(np.sqrt(np.mean((zp - zt) ** 2)))
    rmse_d = float(np.sqrt(np.mean((dp - dt) ** 2)))
    return {
        "rmse_zeta_over_f": rmse_z,
        "r_zeta": r(zp, zt),
        "rmse_div_over_f": rmse_d,
        "r_div": r(dp, dt),
        "std_zeta_over_f_true": float(zt.std()),
        "std_div_over_f_true": float(dt.std()),
        "nrmse_zeta": rmse_z / float(zt.std()),
        "nrmse_div": rmse_d / float(dt.std()),
        # dimensional equivalents [s^-1]
        "rmse_zeta": rmse_z * abs(f),
        "rmse_div": rmse_d * abs(f),
    }


def results_path(results_dir: str, model: PaperModel) -> str:
    return os.path.join(results_dir, model.exp_dir, "checkpoint_best.pt")
