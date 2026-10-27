#!/usr/bin/env python3
"""U-Net architecture diagram (paper Figure 2).

Block labels follow ssh2uv.models.UNet: enc1..enc4 at 64, 32, 16, 8 px with
B, 2B, 4B, 8B channels; bottleneck 16B at 4 px; dec4..dec1 mirror the encoder,
so dec1 (B channels, 64 px) feeds the 1x1 output convolution.

Usage:
    python scripts/figures/fig_unet.py --out figures/unet_architecture
"""

import argparse

from matplotlib.patches import FancyBboxPatch

from ssh2uv.plotting import TEXT_WIDTH_IN, paper_style, plt, save

C_ENC, E_EC = "#BBDEFB", "#1565C0"
C_DEC, D_EC = "#C8E6C9", "#2E7D32"
C_BOT, B_EC = "#FFE0B2", "#E65100"
C_IO, I_EC = "#ECEFF1", "#455A64"
C_SKIP = "#78909C"
C_FLOW = "#263238"


def block(ax, cx, cy, w, h, title, sub, fc, ec):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.04",
                                fc=fc, ec=ec, lw=1.2, zorder=3))
    ax.text(cx, cy + 0.15, title, ha="center", va="center", fontsize=9.5, fontweight="bold", zorder=4)
    ax.text(cx, cy - 0.17, sub, ha="center", va="center", fontsize=8.5, color="#37474F", zorder=4)


def arrow(ax, x1, y1, x2, y2, col=C_FLOW, dashed=False, lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.3", color=col, lw=lw,
                                linestyle=(0, (4, 2.5)) if dashed else "-", shrinkA=0, shrinkB=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/unet_architecture")
    args = p.parse_args()
    paper_style()

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.1))
    ax.set_xlim(0, 13)
    ax.set_ylim(0.2, 8.55)
    ax.set_aspect("equal")
    ax.axis("off")

    bw, bh = 2.55, 0.78
    ex, dxc, bx = 2.85, 10.15, 6.5
    ys = [6.4, 5.15, 3.9, 2.65]
    yb = 1.4
    yio = 7.85
    sizes = [64, 32, 16, 8]
    chans = ["B", "2B", "4B", "8B"]

    block(ax, ex, yio, bw, bh, "Input", r"$C_\mathrm{in}$ × 64×64", C_IO, I_EC)
    block(ax, dxc, yio, bw, bh, r"Output $(u, v)$", "2 × 64×64", C_IO, I_EC)
    for i, (y, n, c) in enumerate(zip(ys, sizes, chans)):
        block(ax, ex, y, bw, bh, f"Enc {i + 1}", f"${c}$ × {n}×{n}", C_ENC, E_EC)
        block(ax, dxc, y, bw, bh, f"Dec {i + 1}", f"${c}$ × {n}×{n}", C_DEC, D_EC)
    block(ax, bx, yb, bw + 0.4, bh, "Bottleneck", r"$16B$ × 4×4", C_BOT, B_EC)

    gap = 0.06
    arrow(ax, ex, yio - bh / 2 - gap, ex, ys[0] + bh / 2 + gap)
    arrow(ax, dxc, ys[0] + bh / 2 + gap, dxc, yio - bh / 2 - gap)
    ax.text(dxc + 0.15, (yio + ys[0]) / 2, "1×1 conv", ha="left", va="center", fontsize=8.5)
    for i in range(3):
        arrow(ax, ex, ys[i] - bh / 2 - gap, ex, ys[i + 1] + bh / 2 + gap)
        arrow(ax, dxc, ys[i + 1] + bh / 2 + gap, dxc, ys[i] - bh / 2 - gap)
    arrow(ax, ex, ys[3] - bh / 2 - gap, bx - (bw + 0.4) / 2 - gap, yb)
    arrow(ax, bx + (bw + 0.4) / 2 + gap, yb, dxc, ys[3] - bh / 2 - gap)
    ax.text(ex - bw / 2 - 0.2, (ys[1] + ys[2]) / 2, "max-pool\n2×2\n(↓)", ha="right", va="center",
            fontsize=8.5)
    ax.text(dxc + bw / 2 + 0.2, (ys[1] + ys[2]) / 2, "transposed\nconv 2×2\n(↑)", ha="left",
            va="center", fontsize=8.5)
    for y in ys:
        arrow(ax, ex + bw / 2 + gap, y, dxc - bw / 2 - gap, y, col=C_SKIP, dashed=True, lw=1.1)
    ax.text(bx, ys[0] + 0.12, "skip: copy and concatenate", ha="center", va="bottom", fontsize=8.5,
            color="#455A64", style="italic")
    ax.text(bx, yb - bh / 2 - 0.4, "each block: (3×3 conv → BN → ReLU) ×2", ha="center", va="center",
            fontsize=8.5, color="#37474F")

    save(fig, args.out)


if __name__ == "__main__":
    main()
