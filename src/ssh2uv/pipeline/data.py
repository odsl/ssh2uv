"""Temporal-split data held as stitched domains, with fixed-tile or random-crop sampling."""

import json
import os
from typing import Dict, Iterator, Tuple

import numpy as np
import torch

from ..kinematics import N_PATCH_COLS, N_PATCH_ROWS, N_PATCHES, PATCH, stitch_patches

DATA_FILE = "global-final-64_stacked_DEEP.npy"  # channels: Eta, windx, windy, theta, u1, v1


def load_split_hours(split_file: str) -> Dict[str, Tuple[int, int]]:
    with open(split_file) as f:
        return {k: tuple(v) for k, v in json.load(f)["split"]["hours"].items()}


def load_domains(data_dir: str, hours: Tuple[int, int]) -> torch.Tensor:
    """Stitched raw fields for hours [a, b): float32 tensor (T, 6, 192, 128)."""
    raw = np.load(os.path.join(data_dir, DATA_FILE), mmap_mode="r")
    a, b = hours
    out = np.empty((b - a, 6, N_PATCH_ROWS * PATCH, N_PATCH_COLS * PATCH), np.float32)
    chunk = 256
    for s in range(a, b, chunk):
        e = min(b, s + chunk)
        block = np.asarray(raw[s * N_PATCHES:e * N_PATCHES]).reshape(e - s, N_PATCHES, 6, PATCH, PATCH)
        out[s - a:e - a] = np.stack([stitch_patches(block[i]) for i in range(e - s)])
    return torch.from_numpy(out)


TILE_ORIGINS = [(r * PATCH, c * PATCH) for r in range(N_PATCH_ROWS) for c in range(N_PATCH_COLS)]


def fixed_tiles(domains: torch.Tensor, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Tiles p (storage order) at hours t: (B, 6, 64, 64)."""
    origins = torch.tensor(TILE_ORIGINS)[p.cpu()]
    return random_crops(domains, t, origins[:, 0], origins[:, 1])


def random_crops(domains: torch.Tensor, t: torch.Tensor, r0: torch.Tensor, c0: torch.Tensor) -> torch.Tensor:
    """64x64 windows with top-left corners (r0, c0) at hours t: (B, 6, 64, 64)."""
    ar = torch.arange(PATCH, device=domains.device)
    rows = (r0.to(domains.device)[:, None] + ar)[:, None, :, None]          # (B,1,64,1)
    cols = (c0.to(domains.device)[:, None] + ar)[:, None, None, :]          # (B,1,1,64)
    d = domains[t.to(domains.device)]                                       # (B,6,192,128)
    ch = torch.arange(d.shape[1], device=domains.device)[None, :, None, None]
    b = torch.arange(d.shape[0], device=domains.device)[:, None, None, None]
    return d[b, ch, rows, cols]


class TrainSampler:
    """Yields raw training batches: fixed tiles (as in the original data) or random crops."""

    def __init__(self, domains: torch.Tensor, batch_size: int, crops: bool, seed: int):
        self.d, self.bs, self.crops = domains, batch_size, crops
        self.g = torch.Generator().manual_seed(seed)
        self.n_hours = domains.shape[0]
        self.steps = (self.n_hours * N_PATCHES) // batch_size  # one epoch = one pass worth of tiles

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.crops:
            for _ in range(self.steps):
                t = torch.randint(0, self.n_hours, (self.bs,), generator=self.g)
                r0 = torch.randint(0, self.d.shape[2] - PATCH + 1, (self.bs,), generator=self.g)
                c0 = torch.randint(0, self.d.shape[3] - PATCH + 1, (self.bs,), generator=self.g)
                yield random_crops(self.d, t, r0, c0)
        else:
            perm = torch.randperm(self.n_hours * N_PATCHES, generator=self.g)
            for s in range(self.steps):
                idx = perm[s * self.bs:(s + 1) * self.bs]
                yield fixed_tiles(self.d, idx // N_PATCHES, idx % N_PATCHES)


def eval_batches(domains: torch.Tensor, batch_size: int) -> Iterator[torch.Tensor]:
    """All fixed tiles of all hours, in storage order (hour-major)."""
    n = domains.shape[0] * N_PATCHES
    idx = torch.arange(n)
    for s in range(0, n, batch_size):
        i = idx[s:s + batch_size]
        yield fixed_tiles(domains, i // N_PATCHES, i % N_PATCHES)
