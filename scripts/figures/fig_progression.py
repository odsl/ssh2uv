#!/usr/bin/env python3
"""Test error of the temporal-split experiments, from baseline to final model (paper Figure 3).

Dots: mean test RMSE of speed over seeds (error bars: one standard deviation);
dashed line: the f-plane geostrophic estimate.

Usage:
    python scripts/figures/fig_progression.py --runs results/temporal --out figures/progression
"""

import argparse
import glob
import json
import os

import numpy as np

from ssh2uv.plotting import TEXT_WIDTH_IN, paper_style, plt, save

STEPS = [
    ("E0_baseline", "baseline (no augmentation)"),
    ("E1_rot", "+ 90° rotations"),
    ("E2_crops", "+ random crops"),
    ("E3_rot_crops", "+ rotations and crops"),
    ("E4_residual", "+ correction to geostrophy"),
    ("R3b_mse_phys", "+ vorticity/divergence penalties"),
    ("C1_bc128_phys", "+ width 128"),
    ("C3_bc128_phys_long", "+ longer training (final)"),
]
SEQ, SEQ_DARK, GRAY, INK, INK2 = "#86b6ef", "#1c5cab", "#8a8985", "#0b0b0b", "#52514e"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/temporal")
    p.add_argument("--temporal_metrics", default="results/temporal_retrain_metrics.json")
    p.add_argument("--out", default="figures/progression")
    a = p.parse_args()
    paper_style()
    geo = json.load(open(a.temporal_metrics))["geostrophic"]["test"]["rmse_speed"]
    rows = []
    for key, label in STEPS:
        v = np.array([json.load(open(f))["test"]["rmse_speed"]
                      for f in sorted(glob.glob(os.path.join(a.runs, key, "seed*", "results.json")))])
        rows.append((label, v.mean(), v.std(ddof=1)))

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.9))
    fig.subplots_adjust(left=0.33, right=0.97, top=0.97, bottom=0.17)
    y = np.arange(len(rows))[::-1]
    colors = ["#3987e5"] * (len(rows) - 1) + [SEQ_DARK]
    ax.errorbar([r[1] for r in rows], y, xerr=[r[2] for r in rows], fmt="none", ecolor=INK2, lw=1, capsize=2.5)
    ax.scatter([r[1] for r in rows], y, s=[40] * (len(rows) - 1) + [70], c=colors, zorder=3,
               edgecolors="white", linewidths=1.2)
    for yi, r in zip(y, rows):
        ax.text(r[1] + r[2] + 0.0007, yi, f"{r[1]:.4f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(geo, color=GRAY, ls="--", lw=1.5)
    ax.text(geo, y[0] + 0.55, f"geostrophic {geo:.4f}", ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0.04, 0.075)
    ax.set_xlabel("test RMSE of speed (m s$^{-1}$), mean ± s.d. over 3 seeds")
    ax.grid(axis="both", color="#e6e5e1", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, a.out)


if __name__ == "__main__":
    main()
