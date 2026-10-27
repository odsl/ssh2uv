"""Training losses: masked MSE/MAE, physics (vorticity/divergence) residuals and direction loss.

Note: the vorticity/divergence residuals here are computed per grid cell
(no division by the grid spacing), exactly as used to train the released models.
Use :mod:`ssh2uv.kinematics` for physically scaled diagnostics.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE loss for (B, C, H, W) tensors with mask (B, 1, H, W).
    
    IMPORTANT: mask must have shape (B, 1, H, W) with a single channel.
    The denominator computation assumes the same mask applies to all output channels.
    """
    assert mask.shape[1] == 1, f"mask must have shape (B, 1, H, W), got {mask.shape}"
    diff2 = (pred.float() - target.float()) ** 2
    diff2 = diff2 * mask.float()
    # mask.sum() counts valid spatial locations (B*1*H*W), multiply by C_out for total elements
    denom = mask.sum() * pred.shape[1] + 1e-8
    return diff2.sum() / denom


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked MAE (L1) loss for (B, C, H, W) tensors with mask (B, 1, H, W)."""
    assert mask.shape[1] == 1, f"mask must have shape (B, 1, H, W), got {mask.shape}"
    diff = (pred.float() - target.float()).abs()
    diff = diff * mask.float()
    denom = mask.sum() * pred.shape[1] + 1e-8
    return diff.sum() / denom


# =============================================================================
# Physics-Informed Spatial Gradient Utilities (for vorticity / divergence loss)
# =============================================================================

def _grad_x(field: torch.Tensor) -> torch.Tensor:
    """Central finite difference in x (column / j-direction) with replicated edges.

    Args:
        field: (B, H, W) tensor
    Returns:
        (B, H, W) tensor  — (f[j+1] - f[j-1]) / 2
    """
    # Replicate-pad left and right columns: (B, H, W+2)
    padded = F.pad(field.unsqueeze(1), (1, 1, 0, 0), mode='replicate').squeeze(1)
    return (padded[:, :, 2:] - padded[:, :, :-2]) * 0.5


def _grad_y(field: torch.Tensor) -> torch.Tensor:
    """Central finite difference in y (row / i-direction) with replicated edges.

    Args:
        field: (B, H, W) tensor
    Returns:
        (B, H, W) tensor  — (f[i+1] - f[i-1]) / 2
    """
    # Replicate-pad top and bottom rows: (B, H+2, W)
    padded = F.pad(field.unsqueeze(1), (0, 0, 1, 1), mode='replicate').squeeze(1)
    return (padded[:, 2:, :] - padded[:, :-2, :]) * 0.5


class VelocityPhysicsLoss(nn.Module):
    """Physics-informed velocity loss: base_loss + a1*vorticity + a2*divergence.

    Algorithm (on uniform 2 km grid):
    ──────────────────────────────────────────────────────────────────────────
    Given predicted (u_pred, v_pred) and target (u_true, v_true) in physical
    units on an H×W uniform grid with spacing dx = dy = 2 km:

    1. Compute vorticity using centered finite differences:
           ζ = ∂v/∂x - ∂u/∂y
       where ∂f/∂x ≈ (f[j+1] - f[j-1]) / (2·dx)
             ∂f/∂y ≈ (f[i+1] - f[i-1]) / (2·dy)
       The factor 1/(2·dx) cancels out in the vorticity *difference* loss
       because we compare ζ_pred vs ζ_true, so only the grid-spacing ratio
       matters (which is 1 for uniform grids).

    2. Compute divergence:
           D = ∂u/∂x + ∂v/∂y

    3. Total loss:
           L = base_loss(pred_norm, target_norm, mask)
             + a1 · MAE(ζ_pred - ζ_true, valid pixels)
             + a2 · MAE(D_pred  - D_true,  valid pixels)

       Base loss operates in *normalised* space for numerical stability.
       Vorticity and divergence are computed in *physical* space (m/s per
       grid cell) by first denormalising the predictions and targets.
    ──────────────────────────────────────────────────────────────────────────

    Args:
        loss_type:   'mse' (L2, default) or 'mae' (L1)
        vort_weight: a1 — weight for the vorticity residual term
        div_weight:  a2 — weight for the divergence residual term
    """

    def __init__(self, loss_type: str = 'mse', vort_weight: float = 0.0,
                 div_weight: float = 0.0):
        super().__init__()
        assert loss_type in ('mse', 'mae'), f"loss_type must be 'mse' or 'mae', got {loss_type}"
        self.loss_type = loss_type
        self.vort_weight = vort_weight
        self.div_weight = div_weight

    def forward(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        mask: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total loss and its three components.

        Args:
            pred_norm:   (B, 2, H, W) normalised predictions [u, v]
            target_norm: (B, 2, H, W) normalised targets [u, v]
            mask:        (B, 1, H, W) validity mask
            y_mean:      (1, 2, 1, 1) output normalisation mean
            y_std:       (1, 2, 1, 1) output normalisation std

        Returns:
            (total_loss, base_loss, vort_loss, div_loss)
        """
        # --- base loss (normalised space) ---
        if self.loss_type == 'mae':
            base = masked_mae(pred_norm, target_norm, mask)
        else:
            base = masked_mse(pred_norm, target_norm, mask)

        if self.vort_weight == 0.0 and self.div_weight == 0.0:
            zero = pred_norm.new_zeros(1)
            return base, base, zero, zero

        # --- denormalise to physical units ---
        pred_phys = pred_norm.float() * y_std + y_mean   # (B, 2, H, W)
        targ_phys = target_norm.float() * y_std + y_mean

        u_p, v_p = pred_phys[:, 0], pred_phys[:, 1]  # (B, H, W)
        u_t, v_t = targ_phys[:, 0], targ_phys[:, 1]

        valid = mask[:, 0]  # (B, H, W)
        n_valid = valid.sum() + 1e-8

        # --- vorticity: ζ = dv/dx - du/dy ---
        vort_pred = _grad_x(v_p) - _grad_y(u_p)
        vort_targ = _grad_x(v_t) - _grad_y(u_t)
        vort_loss = ((vort_pred - vort_targ).abs() * valid).sum() / n_valid

        # --- divergence: D = du/dx + dv/dy ---
        div_pred = _grad_x(u_p) + _grad_y(v_p)
        div_targ = _grad_x(u_t) + _grad_y(v_t)
        div_loss = ((div_pred - div_targ).abs() * valid).sum() / n_valid

        total = base + self.vort_weight * vort_loss + self.div_weight * div_loss
        return total, base, vort_loss, div_loss


def direction_loss_cosine(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    speed_eps: float = 0.01,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Direction loss based on 1 - cos(theta) in physical units."""
    # Ensure y_mean/y_std are on the same device as predictions
    y_mean = y_mean.to(pred_norm.device)
    y_std = y_std.to(pred_norm.device)
    
    # Denormalize to physical units
    pred = pred_norm.float() * y_std + y_mean
    targ = target_norm.float() * y_std + y_mean

    u_p, v_p = pred[:, 0], pred[:, 1]
    u_t, v_t = targ[:, 0], targ[:, 1]

    dot = u_p * u_t + v_p * v_t
    norm_p = torch.sqrt(u_p ** 2 + v_p ** 2 + eps)
    norm_t = torch.sqrt(u_t ** 2 + v_t ** 2 + eps)

    cos = torch.clamp(dot / (norm_p * norm_t + eps), -1.0 + 1e-6, 1.0 - 1e-6)
    dir_err = 1.0 - cos

    # Apply masks
    valid = mask[:, 0] * (norm_t > speed_eps).float()
    return (dir_err * valid).sum() / (valid.sum() + 1e-8)
