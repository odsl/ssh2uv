import math

import numpy as np
import torch

from ssh2uv.kinematics import (
    DX_M, F0, G, LAT0, N_PATCH_COLS, N_PATCH_ROWS, PATCH, divergence, geostrophic_velocity,
    stitch_patches, strain, vorticity,
)

H = W = 32
y, x = torch.meshgrid(torch.arange(H, dtype=torch.float64) * DX_M,
                      torch.arange(W, dtype=torch.float64) * DX_M, indexing="ij")
interior = (slice(1, -1), slice(1, -1))


def test_f0_matches_mean_latitude():
    assert math.isclose(F0, 2 * 7.2921e-5 * math.sin(math.radians(LAT0)))
    assert F0 < 0  # Southern Hemisphere


def test_solid_body_rotation():
    omega = 1e-5
    u, v = -omega * y, omega * x
    assert torch.allclose(vorticity(u, v)[interior], torch.full((H - 2, W - 2), 2 * omega, dtype=torch.float64))
    assert torch.allclose(divergence(u, v)[interior], torch.zeros(H - 2, W - 2, dtype=torch.float64), atol=1e-15)
    assert torch.allclose(strain(u, v)[interior], torch.zeros(H - 2, W - 2, dtype=torch.float64), atol=1e-15)


def test_pure_divergence_and_strain():
    a = 2e-6
    u, v = a * x, a * y
    assert torch.allclose(divergence(u, v)[interior], torch.full((H - 2, W - 2), 2 * a, dtype=torch.float64))
    u, v = a * x, -a * y  # pure normal strain
    assert torch.allclose(strain(u, v)[interior], torch.full((H - 2, W - 2), 2 * a, dtype=torch.float64))


def test_geostrophic_velocity_is_nondivergent_and_signed():
    eta = 0.1 * torch.sin(2 * math.pi * x / (W * DX_M)) * torch.cos(2 * math.pi * y / (H * DX_M))
    ug, vg = geostrophic_velocity(eta)
    assert torch.abs(divergence(ug, vg)[2:-2, 2:-2]).max() < 1e-12
    # eta increasing eastward with f < 0 gives southward flow: v_g = (g/f) d(eta)/dx < 0
    _, vg = geostrophic_velocity(1e-3 * x)
    assert torch.allclose(vg[interior], torch.full_like(vg[interior], G / F0 * 1e-3))  # edges are one-sided halves


def test_cyclonic_is_positive_zeta_over_f():
    # Clockwise rotation (negative zeta) is cyclonic in the Southern Hemisphere.
    omega = 1e-5
    zeta = vorticity(omega * y, -omega * x)[interior]
    assert torch.all(zeta / F0 > 0)


def test_stitch_patches_layout():
    patches = np.arange(N_PATCH_ROWS * N_PATCH_COLS)[:, None, None] * np.ones((1, PATCH, PATCH))
    dom = stitch_patches(patches)
    assert dom.shape == (N_PATCH_ROWS * PATCH, N_PATCH_COLS * PATCH)
    assert dom[0, 0] == 0 and dom[0, -1] == 1          # southern row: patches 0, 1
    assert dom[-1, 0] == 4 and dom[-1, -1] == 5        # northern row: patches 4, 5
