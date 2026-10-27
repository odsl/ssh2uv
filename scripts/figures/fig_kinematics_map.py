#!/usr/bin/env python3
"""Vorticity and divergence maps for one validation patch (paper Figure 4).

Rows: zeta/f and (div u)/f (signed f, so zeta/f > 0 is cyclonic).
Columns: LLC4320 truth | f-plane geostrophic estimate from SSH | U-Net prediction.
The patch shown is the test sample whose per-patch r(zeta) for the U-Net is
closest to the median over the test block, so it is representative rather than
a best case (paper Figure 6). Reads the arrays written by scripts/predict.py and
writes the chosen sample to <out>_sample.json for fig_predictions.py.

Usage:
    python scripts/figures/fig_kinematics_map.py --eval_dir outputs/test_predictions --model c3
"""

import argparse
import json
import os

import numpy as np
import torch

from ssh2uv.evaluation import PAPER_MODELS_BY_KEY
from ssh2uv.kinematics import DX_M, F0, divergence, vorticity
from ssh2uv.plotting import TEXT_WIDTH_IN, panel_label, paper_style, plt, row_colorbar, save, \
    scale_bar, symmetric_limit


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_dir", default="outputs/test_predictions")
    p.add_argument("--model", default="c3", help="prediction key in <eval_dir>/predictions")
    p.add_argument("--label", default="final model", help="Panel title for the model")
    p.add_argument("--sample", type=int, default=None, help="Validation sample (default: median r)")
    p.add_argument("--out", default="figures/kinematics_map")
    return p.parse_args()


def kin(uv):
    t = torch.from_numpy(np.ascontiguousarray(uv)).double()
    return (vorticity(t[:, 0], t[:, 1]).numpy() / F0, divergence(t[:, 0], t[:, 1]).numpy() / F0)


def per_sample_r(a, b):
    a = a.reshape(len(a), -1) - a.reshape(len(a), -1).mean(1, keepdims=True)
    b = b.reshape(len(b), -1) - b.reshape(len(b), -1).mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))


def r(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    args = parse_args()
    paper_style()
    truth = np.load(os.path.join(args.eval_dir, "truth.npy"), mmap_mode="r")
    pred = np.load(os.path.join(args.eval_dir, "predictions", f"{args.model}.npy"), mmap_mode="r")
    geo = np.load(os.path.join(args.eval_dir, "predictions", "geostrophic.npy"), mmap_mode="r")
    val_idx = np.load(os.path.join(args.eval_dir, "val_indices.npy"))

    if args.sample is None:
        zt_all, _ = kin(truth)
        zp_all, _ = kin(pred)
        rs = per_sample_r(zp_all, zt_all)
        args.sample = int(np.argmin(np.abs(rs - np.median(rs))))
        print(f"per-patch r(zeta): median {np.median(rs):.3f}, IQR "
              f"{np.percentile(rs, 25):.3f}-{np.percentile(rs, 75):.3f}")
    k = args.sample
    zt, dt = (a[0] for a in kin(truth[k:k + 1]))
    zg, dg = (a[0] for a in kin(geo[k:k + 1]))
    zp, dp = (a[0] for a in kin(pred[k:k + 1]))
    stats = {"sample": k, "dataset_index": int(val_idx[k]),
             "r_zeta_geo": r(zg, zt), "r_zeta_ml": r(zp, zt), "r_div_ml": r(dp, dt)}
    print(json.dumps(stats))

    label = args.label or PAPER_MODELS_BY_KEY[args.model].label
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH_IN, 4.45))
    fig.subplots_adjust(left=0.07, right=0.865, top=0.9, bottom=0.02, wspace=0.06, hspace=0.08)
    titles = ["LLC4320 (truth)", "Geostrophic", f"U-Net ({label})"]
    lim_z = symmetric_limit(zt)
    lim_d = symmetric_limit(dt)
    kw = dict(origin="lower", interpolation="nearest")

    rows = [
        ([zt, zg, zp], "RdBu_r", lim_z, "Vorticity", r"$\zeta/f$  (> 0 cyclonic)",
         [None, r(zg, zt), r(zp, zt)]),
        ([dt, None, dp], "PuOr_r", lim_d, "Divergence", r"$(\nabla\!\cdot\!\mathbf{u})/f$",
         [None, None, r(dp, dt)]),
    ]
    for i, (fields, cmap, lim, name, cbar_label, rvals) in enumerate(rows):
        im = None
        for j, data in enumerate(fields):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if data is None:
                ax.set_facecolor("0.96")
                ax.set_xlim(-0.5, zt.shape[1] - 0.5)
                ax.set_ylim(-0.5, zt.shape[0] - 0.5)
                ax.set_aspect("equal")
                ax.text(0.5, 0.5, r"$\nabla\!\cdot\!\mathbf{u}_g \equiv 0$", transform=ax.transAxes,
                        ha="center", va="center", fontsize=11)
            else:
                im = ax.imshow(data, cmap=cmap, vmin=-lim, vmax=lim, **kw)
            if rvals[j] is not None:
                ax.text(0.97, 0.03, f"$r = {rvals[j]:.2f}$", transform=ax.transAxes, ha="right",
                        va="bottom", fontsize=9.5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
            panel_label(ax, "abcdef"[3 * i + j])
            if i == 0:
                ax.set_title(titles[j])
        axes[i, 0].set_ylabel(name, fontsize=11)
        ext = "both" if np.abs(fields[1] if fields[1] is not None else 0).max() > lim or \
            np.abs(fields[0]).max() > lim else "neither"
        row_colorbar(fig, im, axes[i], cbar_label, extend=ext)
    scale_bar(axes[1, 0], 50, DX_M / 1000.0, loc=(0.08, 0.08))

    save(fig, args.out)
    with open(args.out + "_sample.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
