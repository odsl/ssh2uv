"""Physical constants, domain geometry and kinematic diagnostics.

All diagnostics use centred differences on the regular 2 km analysis grid with
replicated edges (one-sided half differences on the outermost row/column),
the same stencil used by the training losses in :mod:`ssh2uv.losses`.

Sign convention
---------------
The domain is in the Southern Hemisphere, so ``F0 < 0``. Vorticity is
reported as the signed ratio ``zeta / f``: ``zeta / f > 0`` is cyclonic and
``zeta / f < 0`` anticyclonic in either hemisphere. Strain and other
non-negative quantities are normalised by ``|f|``.
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

G = 9.81                    # gravitational acceleration [m s^-2]
OMEGA = 7.2921e-5           # Earth's rotation rate [s^-1]

# Analysis grid (scripts/preprocess.py regrids LLC4320 to a regular 2 km grid)
DX_M = 2000.0               # grid spacing [m], dx == dy

# Pre-SWOT LLC4320 "Cape Basin" region (read from the granule XC/YC coordinates)
DOMAIN_LON = (11.82, 15.80)  # degrees east
DOMAIN_LAT = (-44.99, -41.01)  # degrees north
LAT0 = -43.0                # mean latitude of the region [deg]
F0 = 2.0 * OMEGA * math.sin(math.radians(LAT0))  # f-plane Coriolis parameter [s^-1], ~ -9.95e-5

# Patch layout written by scripts/preprocess.py: six 64x64 patches on a
# 3 (south->north) x 2 (west->east) grid, stored row-major, sample = t * 6 + p
PATCH = 64
N_PATCH_ROWS = 3
N_PATCH_COLS = 2
N_PATCHES = N_PATCH_ROWS * N_PATCH_COLS
KM_PER_DEG_LAT = 111.2


# ---------------------------------------------------------------------------
# Finite differences
# ---------------------------------------------------------------------------

def _as_batch(field: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if field.dim() == 2:
        return field.unsqueeze(0), True
    return field, False


def ddx(field: torch.Tensor, dx: float = DX_M) -> torch.Tensor:
    """d/dx along the last (column, eastward) axis of a (B, H, W) or (H, W) tensor."""
    f, squeeze = _as_batch(field)
    padded = F.pad(f.unsqueeze(1), (1, 1, 0, 0), mode="replicate").squeeze(1)
    out = (padded[:, :, 2:] - padded[:, :, :-2]) / (2.0 * dx)
    return out.squeeze(0) if squeeze else out


def ddy(field: torch.Tensor, dx: float = DX_M) -> torch.Tensor:
    """d/dy along the row (northward) axis of a (B, H, W) or (H, W) tensor."""
    f, squeeze = _as_batch(field)
    padded = F.pad(f.unsqueeze(1), (0, 0, 1, 1), mode="replicate").squeeze(1)
    out = (padded[:, 2:, :] - padded[:, :-2, :]) / (2.0 * dx)
    return out.squeeze(0) if squeeze else out


# ---------------------------------------------------------------------------
# Kinematic quantities
# ---------------------------------------------------------------------------

def vorticity(u: torch.Tensor, v: torch.Tensor, dx: float = DX_M) -> torch.Tensor:
    """Relative vorticity zeta = dv/dx - du/dy [s^-1]."""
    return ddx(v, dx) - ddy(u, dx)


def divergence(u: torch.Tensor, v: torch.Tensor, dx: float = DX_M) -> torch.Tensor:
    """Horizontal divergence du/dx + dv/dy [s^-1]."""
    return ddx(u, dx) + ddy(v, dx)


def strain(u: torch.Tensor, v: torch.Tensor, dx: float = DX_M) -> torch.Tensor:
    """Total strain rate sqrt((du/dx - dv/dy)^2 + (du/dy + dv/dx)^2) [s^-1]."""
    normal = ddx(u, dx) - ddy(v, dx)
    shear = ddy(u, dx) + ddx(v, dx)
    return torch.sqrt(normal ** 2 + shear ** 2)


def geostrophic_velocity(
    eta: torch.Tensor, f: float = F0, dx: float = DX_M, g: float = G
) -> Tuple[torch.Tensor, torch.Tensor]:
    """f-plane geostrophic velocity from SSH: u_g = -(g/f) deta/dy, v_g = (g/f) deta/dx."""
    return -(g / f) * ddy(eta, dx), (g / f) * ddx(eta, dx)


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

def stitch_patches(patches):
    """Assemble the six patches of one time step into the (192, 128) domain.

    Args:
        patches: array-like of shape (6, ..., 64, 64) in storage order.
    Returns:
        array of shape (..., 192, 128); row 0 is the southern edge.
    """
    import numpy as np

    patches = np.asarray(patches)
    rows = [
        np.concatenate([patches[r * N_PATCH_COLS + c] for c in range(N_PATCH_COLS)], axis=-1)
        for r in range(N_PATCH_ROWS)
    ]
    return np.concatenate(rows, axis=-2)


def y_km_to_lat(y_km):
    """Latitude [deg] of a northward distance from the southern domain edge."""
    return DOMAIN_LAT[0] + y_km / KM_PER_DEG_LAT


def lat_to_y_km(lat):
    return (lat - DOMAIN_LAT[0]) * KM_PER_DEG_LAT
