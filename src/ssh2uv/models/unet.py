"""U-Net encoder-decoder used for all SSH-to-velocity experiments.

Attribute names must stay unchanged so that existing checkpoints
(``ckpt["model_state"]``) load with ``strict=True``.
"""

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two-layer conv block with BatchNorm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net architecture for variable input sizes (must be divisible by 16)."""

    def __init__(self, in_channels: int, out_channels: int = 2, base: int = 32):
        super().__init__()
        
        feats = [base, base * 2, base * 4, base * 8, base * 16]

        # Encoder
        self.enc1 = conv_block(in_channels, feats[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(feats[0], feats[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(feats[1], feats[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(feats[2], feats[3])
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(feats[3], feats[4])

        # Decoder
        self.up4 = nn.ConvTranspose2d(feats[4], feats[3], kernel_size=2, stride=2)
        self.dec4 = conv_block(feats[3] + feats[3], feats[3])
        self.up3 = nn.ConvTranspose2d(feats[3], feats[2], kernel_size=2, stride=2)
        self.dec3 = conv_block(feats[2] + feats[2], feats[2])
        self.up2 = nn.ConvTranspose2d(feats[2], feats[1], kernel_size=2, stride=2)
        self.dec2 = conv_block(feats[1] + feats[1], feats[1])
        self.up1 = nn.ConvTranspose2d(feats[1], feats[0], kernel_size=2, stride=2)
        self.dec1 = conv_block(feats[0] + feats[0], feats[0])

        self.out = nn.Conv2d(feats[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)
