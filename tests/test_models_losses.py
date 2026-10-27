import torch

from ssh2uv.losses import VelocityPhysicsLoss, direction_loss_cosine, masked_mae, masked_mse
from ssh2uv.models import UNet


def test_unet_shapes_and_channel_progression():
    model = UNet(in_channels=5, out_channels=2, base=8)
    out = model(torch.randn(2, 5, 64, 64))
    assert out.shape == (2, 2, 64, 64)
    assert model.bottleneck[0].out_channels == 16 * 8
    assert model.dec1[0].out_channels == 8  # dec1 (B channels) feeds the output conv


def test_losses_vanish_for_perfect_prediction():
    y = torch.randn(3, 2, 16, 16)
    mask = torch.ones(3, 1, 16, 16)
    mean, std = torch.zeros(1, 2, 1, 1), torch.ones(1, 2, 1, 1)
    total, base, vort, div = VelocityPhysicsLoss("mae", 0.1, 0.1)(y, y, mask, mean, std)
    assert total.item() == 0 and vort.item() == 0 and div.item() == 0
    assert masked_mse(y, y, mask).item() == 0
    assert direction_loss_cosine(y, y, mask, mean, std).item() < 1e-5


def test_masked_losses_average_over_valid_pixels_and_channels():
    pred = torch.ones(1, 2, 4, 4)
    target = torch.zeros(1, 2, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., :2, :] = 1
    assert abs(masked_mae(pred, target, mask).item() - 1.0) < 1e-6
    assert abs(masked_mse(2 * pred, target, mask).item() - 4.0) < 1e-6
