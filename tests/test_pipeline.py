import math

import torch

from ssh2uv.kinematics import vorticity
from ssh2uv.pipeline.features import build_inputs, build_target, geostrophic, random_rotate, rotate90, to_velocity

H = W = 32


def smooth_raw(seed=0):
    g = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(torch.linspace(0, 2 * math.pi, H, dtype=torch.float64),
                          torch.linspace(0, 2 * math.pi, W, dtype=torch.float64), indexing="ij")
    a = torch.rand(6, generator=g, dtype=torch.float64)
    eta = 0.3 * torch.sin(x + a[0]) * torch.cos(2 * y + a[1]) + 0.1 * y
    ug, vg = geostrophic(eta[None])
    raw = torch.stack([eta, a[2] * torch.cos(y), a[3] * torch.sin(x), 10 + x, ug[0], vg[0]])[None]
    return raw


def test_rotation_preserves_geostrophy_and_vorticity():
    raw = smooth_raw()
    for k in range(4):
        rot = rotate90(raw, k)
        ug, vg = geostrophic(rot[:, 0])
        inner = (slice(None), slice(2, -2), slice(2, -2))
        # rotated (u, v) must still be the geostrophic velocity of the rotated SSH
        assert torch.allclose(rot[:, 4][inner], ug[inner], atol=1e-9)
        assert torch.allclose(rot[:, 5][inner], vg[inner], atol=1e-9)
        # vorticity is a scalar: rotating the flow rotates its vorticity field
        z_rot = vorticity(rot[:, 4], rot[:, 5])
        z_ref = torch.rot90(vorticity(raw[:, 4], raw[:, 5]), k, dims=(-1, -2))
        assert torch.allclose(z_rot[inner], z_ref[inner], atol=1e-9)


def test_rotation_group_properties():
    raw = smooth_raw(1)
    assert torch.allclose(rotate90(rotate90(raw, 1), 3), raw)
    assert torch.allclose(rotate90(raw, 4), raw)
    # an eastward vector becomes northward after one counter-clockwise rotation
    r = torch.zeros(1, 6, 4, 4, dtype=torch.float64)
    r[:, 4] = 1.0
    out = rotate90(r, 1)
    assert torch.allclose(out[:, 4], torch.zeros(1, 4, 4, dtype=torch.float64))
    assert torch.allclose(out[:, 5], torch.ones(1, 4, 4, dtype=torch.float64))


def test_random_rotate_keeps_samples_consistent():
    raw = torch.cat([smooth_raw(s) for s in range(8)])
    out = random_rotate(raw, torch.Generator().manual_seed(0))
    ug, vg = geostrophic(out[:, 0])
    inner = (slice(None), slice(2, -2), slice(2, -2))
    assert torch.allclose(out[:, 4][inner], ug[inner], atol=1e-9)


def test_residual_target_roundtrip_and_inputs():
    raw = smooth_raw()
    res = build_target(raw, "residual")
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-12)  # field is exactly geostrophic
    assert torch.allclose(to_velocity(res, raw, "residual"), raw[:, 4:6])
    x = build_inputs(raw, ["grad", "sst", "wind", "ug"])
    assert x.shape == (1, 7, H, W)


def test_fixed_tiles_match_stored_patches():
    import numpy as np
    from ssh2uv.kinematics import stitch_patches
    from ssh2uv.pipeline.data import fixed_tiles, random_crops
    rng = np.random.default_rng(0)
    patches = rng.normal(size=(3, 6, 6, 64, 64)).astype(np.float32)   # (hours, patches, ch, 64, 64)
    dom = torch.from_numpy(np.stack([stitch_patches(patches[t]) for t in range(3)]))
    t = torch.tensor([0, 1, 2, 2]); p = torch.tensor([0, 5, 3, 1])
    out = fixed_tiles(dom, t, p)
    for i in range(4):
        assert torch.allclose(out[i], torch.from_numpy(patches[t[i], p[i]]))
    crop = random_crops(dom, torch.tensor([1]), torch.tensor([10]), torch.tensor([20]))
    assert torch.allclose(crop[0], dom[1, :, 10:74, 20:84])


def test_smoothed_geostrophy_roundtrip():
    from ssh2uv.pipeline.features import gaussian_smooth
    raw = smooth_raw()
    for s in (0.0, 2.0):
        res = build_target(raw, "residual", s)
        assert torch.allclose(to_velocity(res, raw, "residual", s), raw[:, 4:6])
    const = torch.full((1, 16, 16), 3.0, dtype=torch.float64)
    assert torch.allclose(gaussian_smooth(const, 2.0), const)
