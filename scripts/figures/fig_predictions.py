#!/usr/bin/env python3
"""Velocity on a representative test patch (paper Figure 4).

Columns: LLC4320 truth | geostrophic estimate | baseline U-Net (E0) | final U-Net (C3)
Rows: u, v (shared color scale per row) and the vector error |u_pred - u_true|.
Uses the same test patch as fig_kinematics_map.py (read from --sample_json).
Reads the arrays written by scripts/predict.py.

Usage:
    python scripts/figures/fig_predictions.py --eval_dir outputs/test_predictions --out figures/predictions
"""

import argparse
import json
import os

import numpy as np

from ssh2uv.kinematics import DX_M
from ssh2uv.plotting import TEXT_WIDTH_IN, panel_label, paper_style, plt, row_colorbar, save, \
    scale_bar, symmetric_limit



def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_dir", default="outputs/test_predictions")
    p.add_argument("--sample", type=int, default=None,
                   help="Validation sample (default: the one chosen for the kinematics map)")
    p.add_argument("--out", default="figures/predictions")
    p.add_argument("--models", nargs="+", default=["geostrophic", "e0", "c3"], help="prediction keys in <eval_dir>/predictions")
    p.add_argument("--labels", nargs="+", default=["Geostrophic", "Baseline U-Net", "Final U-Net"], help="column titles for --models")
    p.add_argument("--sample_json", default="figures/kinematics_map_sample.json")
    return p.parse_args()


def main():
    args = parse_args()
    paper_style()
    if args.sample is None:
        with open(args.sample_json) as f:
            args.sample = json.load(f)["sample"]
    k = args.sample
    truth = np.load(os.path.join(args.eval_dir, "truth.npy"), mmap_mode="r")[k]
    preds = {m: np.load(os.path.join(args.eval_dir, "predictions", f"{m}.npy"), mmap_mode="r")[k]
             for m in args.models}

    fig, axes = plt.subplots(3, 4, figsize=(TEXT_WIDTH_IN, 5.05))
    fig.subplots_adjust(left=0.06, right=0.875, top=0.93, bottom=0.01, wspace=0.05, hspace=0.07)
    labels = args.labels
    titles = ["LLC4320 (truth)"] + labels
    kw = dict(origin="lower", interpolation="nearest")
    lim_u = symmetric_limit(truth[0], 99.5)
    lim_v = symmetric_limit(truth[1], 99.5)
    errs = {m: np.hypot(*(preds[m] - truth)) for m in args.models}
    emax = float(np.percentile(np.concatenate([e.ravel() for e in errs.values()]), 99.5))

    for i, (comp, lim, name) in enumerate([(0, lim_u, "$u$"), (1, lim_v, "$v$")]):
        for j, data in enumerate([truth] + [preds[m] for m in args.models]):
            im = axes[i, j].imshow(data[comp], cmap="RdBu_r", vmin=-lim, vmax=lim, **kw)
        axes[i, 0].set_ylabel(name, fontsize=11)
        row_colorbar(fig, im, axes[i], f"{name} (m s$^{{-1}}$)", extend="both")

    ax = axes[2, 0]
    ax.set_xlim(-0.5, 63.5)
    ax.set_ylim(-0.5, 63.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.5, 0.5, "vector error\n$|\\hat{\\mathbf{u}} - \\mathbf{u}|$\n\nnumbers: speed\nRMSE of this\npatch (m s$^{-1}$)",
            transform=ax.transAxes, ha="center", va="center", fontsize=9)
    for j, m in enumerate(args.models, start=1):
        im = axes[2, j].imshow(errs[m], cmap="Greys", vmin=0, vmax=emax, **kw)
        sp_err = np.hypot(*preds[m]) - np.hypot(*truth)
        axes[2, j].text(0.97, 0.03, f"{np.sqrt(np.mean(sp_err ** 2)):.3f}", transform=axes[2, j].transAxes,
                        ha="right", va="bottom", fontsize=9.5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    axes[2, 0].set_ylabel("error", fontsize=11)
    row_colorbar(fig, im, axes[2, 1:], r"$|\hat{\mathbf{u}} - \mathbf{u}|$ (m s$^{-1}$)", extend="max")

    letters = iter("abcdefghijkl")
    for i in range(3):
        for j in range(4):
            if i == 2 and j == 0:
                continue
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            panel_label(axes[i, j], next(letters))
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=9.5)
    scale_bar(axes[1, 0], 50, DX_M / 1000.0, loc=(0.08, 0.08))
    save(fig, args.out)


if __name__ == "__main__":
    main()
