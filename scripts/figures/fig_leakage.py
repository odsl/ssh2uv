#!/usr/bin/env python3
"""Leakage diagnostics for the random patch split (paper Figure 8, Appendix B).

(a) Validation error of the random-split models as a function of the time to the
    nearest training patch of the same subdomain, with the geostrophic estimate
    (which uses no training data) and a no-physics oracle that interpolates the
    neighbouring training patches in time.
(b) The same model setup (Grad+All, B=192) scored on the random-split validation
    set and on the held-out test block of the temporal split, with geostrophy and
    the final temporal-split model for reference.

Usage:
    python scripts/figures/fig_leakage.py --eval_dir outputs/random_split_eval --out figures/leakage
"""

import argparse
import glob
import json
import os

import numpy as np

from ssh2uv.kinematics import N_PATCHES
from ssh2uv.plotting import TEXT_WIDTH_IN, panel_label, paper_style, plt, save

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2 = "#8a8985", "#0b0b0b", "#52514e"
SEQ_LIGHT, SEQ_DARK = "#86b6ef", "#1c5cab"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_dir", default="outputs/random_split_eval")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--temporal_metrics", default="results/temporal_retrain_metrics.json")
    p.add_argument("--final_runs", default="results/temporal/C3_bc128_phys_long")
    p.add_argument("--out", default="figures/leakage")
    return p.parse_args()


def speed_rmse(a, b):
    return float(np.sqrt(np.mean((np.hypot(a[:, 0], a[:, 1]) - np.hypot(b[:, 0], b[:, 1])) ** 2)))


def main():
    args = parse_args()
    paper_style()
    val = np.load(os.path.join(args.eval_dir, "val_indices.npy"))
    truth = np.load(os.path.join(args.eval_dir, "truth.npy"), mmap_mode="r")
    raw = np.load(os.path.join(args.data_dir, "global-final-64_stacked_DEEP.npy"), mmap_mode="r")
    n = raw.shape[0]
    n_hours = n // N_PATCHES
    is_val = np.zeros(n, bool)
    is_val[val] = True
    t_val, p_val = val // N_PATCHES, val % N_PATCHES

    # gap to nearest training hour (same subdomain) and time-interpolation oracle
    gap = np.empty(len(val), int)
    oracle = np.empty((len(val), 2, 64, 64), np.float32)
    for i, (t, p) in enumerate(zip(t_val, p_val)):
        nb = []
        for step in (-1, 1):
            s = t + step
            while 0 <= s < n_hours and is_val[s * N_PATCHES + p]:
                s += step
            nb.append(s if 0 <= s < n_hours else None)
        (a, b) = nb
        ga = t - a if a is not None else 10 ** 9
        gb = b - t if b is not None else 10 ** 9
        gap[i] = min(ga, gb)
        if a is not None and b is not None:
            w = ga / (ga + gb)
            oracle[i] = (1 - w) * raw[a * N_PATCHES + p, 4:6] + w * raw[b * N_PATCHES + p, 4:6]
        else:
            oracle[i] = raw[(a if a is not None else b) * N_PATCHES + p, 4:6]

    gaps = [1, 2, 3]
    series = [("ssh_all", "SSH+All", ORANGE), ("grad_all", "Grad+All", AQUA), ("grad_all_lg_phys", "Grad+All-Lg/Phys", BLUE)]
    stats = {"n_per_gap": {g: int((gap == g).sum()) for g in gaps}}
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.8), gridspec_kw={"width_ratios": [1.05, 1]})
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.24, wspace=0.38)

    ax = axes[0]
    for key, label, color in series + [("geostrophic", "geostrophic", GRAY)]:
        pred = np.load(os.path.join(args.eval_dir, "predictions", f"{key}.npy"), mmap_mode="r")
        ys = [speed_rmse(np.asarray(pred[gap == g]), np.asarray(truth[gap == g])) for g in gaps]
        stats[label] = ys
        ls = "--" if key == "geostrophic" else "-"
        ax.plot(gaps, ys, ls, color=color, lw=2, marker="o", ms=5, label=label)
    ys = [speed_rmse(oracle[gap == g], np.asarray(truth[gap == g])) for g in gaps]
    stats["oracle"] = ys
    ax.plot(gaps, ys, ":", color=INK, lw=2, marker="s", ms=5, label="time interpolation\n(no physics)")
    ax.set_xticks(gaps)
    ax.set_xlabel("hours to nearest training patch")
    ax.set_ylabel("RMSE of speed (m s$^{-1}$)")
    ax.set_ylim(0, 0.11)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.0, 1.0, 1.08), ncol=2, fontsize=7.5, frameon=False,
              handlelength=2.0, columnspacing=1.0)
    panel_label(ax, "a", loc="upper right")

    # (b) same setup, two evaluation protocols
    tm = json.load(open(args.temporal_metrics))
    finals = [json.load(open(f))["test"]["rmse_speed"] for f in sorted(glob.glob(os.path.join(args.final_runs, "seed*", "results.json")))]
    rand_val = float(np.sqrt(np.mean((np.hypot(*np.load(os.path.join(args.eval_dir, "predictions", "grad_all.npy"), mmap_mode="r")[::1].transpose(1, 0, 2, 3)) - np.hypot(*np.asarray(truth).transpose(1, 0, 2, 3))) ** 2)))
    bars = [("random\nsplit\n(val.)", rand_val, None, SEQ_LIGHT),
            ("temporal\nsplit\n(test)", tm["temporal"]["test"]["rmse_speed"], None, SEQ_LIGHT),
            ("geo-\nstrophic\n(test)", tm["geostrophic"]["test"]["rmse_speed"], None, GRAY),
            ("final\nmodel\n(test)", float(np.mean(finals)), float(np.std(finals, ddof=1)), SEQ_DARK)]
    stats["bars"] = [(b[0].replace("\n", " "), b[1]) for b in bars]
    ax = axes[1]
    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], width=0.62, color=[b[3] for b in bars], edgecolor="white", linewidth=1.5)
    for xi, b in zip(x, bars):
        if b[2]:
            ax.errorbar(xi, b[1], yerr=b[2], fmt="none", lw=1, capsize=2, ecolor=INK2)
    for xi, b in zip(x, bars):
        ax.text(xi, b[1] + 0.002, f"{b[1]:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel("RMSE of speed (m s$^{-1}$)")
    ax.set_ylim(0, 0.085)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    ax.set_axisbelow(True)
    panel_label(ax, "b", loc="upper left")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)

    save(fig, args.out)
    with open(args.out + "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
