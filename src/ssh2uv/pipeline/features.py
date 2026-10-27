"""Augmentation and feature construction on raw (B, 6, H, W) batches.

Raw channel order: eta, tau_x, tau_y, theta, u, v. Arrays are (y, x) with y
northward and x eastward, so a physical counter-clockwise rotation by 90 degrees
moves the x axis onto the y axis.
"""

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..kinematics import DX_M, F0, G, ddx, ddy

RAW = {"eta": 0, "tau_x": 1, "tau_y": 2, "theta": 3, "u": 4, "v": 5}
VECTOR_PAIRS = [(1, 2), (4, 5)]  # (tau_x, tau_y), (u, v)

# Input groups -> number of channels they contribute
INPUT_CHANNELS = {"eta": 1, "grad": 2, "ug": 2, "sst": 1, "wind": 2}


def rotate90(raw: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate a (B, 6, H, W) batch counter-clockwise by k*90 degrees (fields and vectors)."""
    k = k % 4
    if k == 0:
        return raw
    out = torch.rot90(raw, k, dims=(-1, -2))
    for _ in range(k):
        out = out.clone()
        for ix, iy in VECTOR_PAIRS:
            x, y = out[:, ix].clone(), out[:, iy].clone()
            out[:, ix], out[:, iy] = -y, x
    return out


def random_rotate(raw: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
    """Rotate each sample by an independent random multiple of 90 degrees."""
    ks = torch.randint(0, 4, (raw.shape[0],), generator=generator, device="cpu")
    out = raw.clone()
    for k in range(1, 4):
        sel = (ks == k).nonzero(as_tuple=True)[0].to(raw.device)
        if len(sel):
            out[sel] = rotate90(raw[sel], k)
    return out


def geostrophic(eta: torch.Tensor, f: float = F0, dx: float = DX_M) -> Tuple[torch.Tensor, torch.Tensor]:
    """f-plane geostrophic velocity (B, H, W) from SSH (B, H, W)."""
    return -(G / f) * ddy(eta, dx), (G / f) * ddx(eta, dx)


def gaussian_smooth(field: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Isotropic Gaussian smoothing of (B, H, W) fields (replicated edges); sigma in grid cells."""
    if not sigma_px or sigma_px <= 0:
        return field
    r = int(math.ceil(3 * sigma_px))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=field.dtype, device=field.device) / sigma_px) ** 2)
    k = k / k.sum()
    x = F.pad(field[:, None], (r, r, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x[:, 0]


def geostrophic_from_raw(raw: torch.Tensor, smooth_px: float = 0.0) -> torch.Tensor:
    """(B, 2, H, W) geostrophic velocity from the (optionally smoothed) SSH channel."""
    ug, vg = geostrophic(gaussian_smooth(raw[:, RAW["eta"]], smooth_px))
    return torch.stack([ug, vg], dim=1)


def build_inputs(raw: torch.Tensor, inputs: Sequence[str], smooth_px: float = 0.0) -> torch.Tensor:
    """Stack the requested input groups (physical units) into (B, C, H, W)."""
    eta = raw[:, RAW["eta"]]
    chans: List[torch.Tensor] = []
    for name in inputs:
        if name == "eta":
            chans.append(eta)
        elif name == "grad":
            chans += [ddx(eta), ddy(eta)]
        elif name == "ug":
            chans += list(geostrophic_from_raw(raw, smooth_px).unbind(1))
        elif name == "sst":
            chans.append(raw[:, RAW["theta"]])
        elif name == "wind":
            chans += [raw[:, RAW["tau_x"]], raw[:, RAW["tau_y"]]]
        else:
            raise ValueError(f"unknown input group {name!r}")
    return torch.stack(chans, dim=1)


def build_target(raw: torch.Tensor, target: str, smooth_px: float = 0.0) -> torch.Tensor:
    """Training target (B, 2, H, W): the velocity, or its departure from geostrophy."""
    uv = raw[:, RAW["u"]:RAW["v"] + 1]
    if target == "uv":
        return uv
    if target == "residual":
        return uv - geostrophic_from_raw(raw, smooth_px)
    raise ValueError(f"unknown target {target!r}")


def to_velocity(pred_target: torch.Tensor, raw: torch.Tensor, target: str, smooth_px: float = 0.0) -> torch.Tensor:
    """Convert a prediction of the target back to velocity (B, 2, H, W)."""
    if target == "uv":
        return pred_target
    return pred_target + geostrophic_from_raw(raw, smooth_px)


def n_input_channels(inputs: Sequence[str]) -> int:
    return sum(INPUT_CHANNELS[n] for n in inputs)


class Standardizer:
    """Per-channel mean/std, estimated on training samples only."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean, self.std = mean.view(1, -1, 1, 1), std.clamp_min(1e-12).view(1, -1, 1, 1)

    def to(self, device):
        return Standardizer(self.mean.to(device).flatten(), self.std.to(device).flatten())

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean

    def state(self) -> Dict[str, list]:
        return {"mean": self.mean.flatten().tolist(), "std": self.std.flatten().tolist()}

    @classmethod
    def fit(cls, x: torch.Tensor):
        """x: (N, C, H, W) -> statistics over N, H, W."""
        x = x.double()
        return cls(x.mean(dim=(0, 2, 3)).float(), x.std(dim=(0, 2, 3)).float())
