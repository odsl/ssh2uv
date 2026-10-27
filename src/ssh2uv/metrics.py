"""Pointwise velocity metrics (RMSE, MAE, speed RMSE, direction error)."""

import math
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    speed_eps: float,
) -> Dict[str, float]:
    """Compute physical-unit metrics on a data loader.
    
    Note: y_mean and y_std can be on any device - they will be moved to 
    the correct device automatically. This allows calling from standalone
    evaluation scripts.
    """
    model.eval()
    
    # Ensure normalization tensors are on the correct device
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)

    sse_u, sse_v, sae_u, sae_v = 0.0, 0.0, 0.0, 0.0
    sse_speed, sum_dir_deg = 0.0, 0.0
    count, count_dir = 0.0, 0.0

    for x, y_norm, mask in tqdm(loader, desc="Evaluating", leave=False):
        x = x.to(device, non_blocking=True)
        y_norm = y_norm.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        pred_norm = model(x)

        # Denormalize
        pred = pred_norm * y_std + y_mean
        targ = y_norm * y_std + y_mean
        valid = mask[:, 0]

        # Component errors
        du = pred[:, 0] - targ[:, 0]
        dv = pred[:, 1] - targ[:, 1]
        sse_u += ((du ** 2) * valid).sum().item()
        sse_v += ((dv ** 2) * valid).sum().item()
        sae_u += (du.abs() * valid).sum().item()
        sae_v += (dv.abs() * valid).sum().item()
        count += valid.sum().item()

        # Speed error
        sp = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + 1e-8)
        st = torch.sqrt(targ[:, 0] ** 2 + targ[:, 1] ** 2 + 1e-8)
        sse_speed += (((sp - st) ** 2) * valid).sum().item()

        # Direction error
        dot = pred[:, 0] * targ[:, 0] + pred[:, 1] * targ[:, 1]
        npred = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + 1e-8)
        ntarg = torch.sqrt(targ[:, 0] ** 2 + targ[:, 1] ** 2 + 1e-8)
        cos = torch.clamp(dot / (npred * ntarg + 1e-8), -1.0, 1.0)
        ang = torch.acos(cos) * (180.0 / math.pi)

        dir_valid = valid * (ntarg > speed_eps).float()
        sum_dir_deg += (ang * dir_valid).sum().item()
        count_dir += dir_valid.sum().item()

    count = max(count, 1.0)
    count_dir = max(count_dir, 1.0)

    return {
        "rmse_u": math.sqrt(sse_u / count),
        "rmse_v": math.sqrt(sse_v / count),
        "mae_u": sae_u / count,
        "mae_v": sae_v / count,
        "rmse_speed": math.sqrt(sse_speed / count),
        "mean_dir_error_deg": sum_dir_deg / count_dir,
    }
