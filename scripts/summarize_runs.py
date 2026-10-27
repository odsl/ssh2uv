#!/usr/bin/env python3
"""Summarize temporal-split runs: mean +/- std over seeds of the held-out test metrics.

Usage:
    python scripts/summarize_runs.py --runs results/temporal --out results/summary.md
"""

import argparse
import glob
import json
import os

import numpy as np

METRICS = [("rmse_speed", "RMSE speed (m/s)", 4), ("mean_dir_error_deg", "dir. err (deg)", 1),
           ("r_zeta", "r(zeta)", 3), ("rmse_zeta_over_f", "RMSE(zeta/f)", 3), ("r_div", "r(div)", 3)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/temporal", help="results/temporal (shipped) or runs/temporal")
    p.add_argument("--geostrophic", default="results/temporal_retrain_metrics.json")
    p.add_argument("--out", default="results/summary.md")
    args = p.parse_args()

    rows = []
    if os.path.exists(args.geostrophic):
        g = json.load(open(args.geostrophic))["geostrophic"]["test"]
        rows.append(("geostrophic (no training)", "-", "-", {k: f"{g[k]:.{d}f}" for k, _, d in METRICS}))
    for exp in sorted(os.listdir(args.runs)):
        files = sorted(glob.glob(os.path.join(args.runs, exp, "seed*", "results.json")))
        if not files:
            continue
        res = [json.load(open(f)) for f in files]
        cells = {}
        for k, _, d in METRICS:
            v = np.array([r["test"][k] for r in res])
            cells[k] = f"{v.mean():.{d}f} ± {v.std(ddof=1) if len(v) > 1 else 0:.{d}f}"
        epochs = "/".join(str(r["best_epoch"]) for r in res)
        rows.append((exp, str(len(res)), epochs, cells))

    head = "| experiment | seeds | best epochs | " + " | ".join(h for _, h, _ in METRICS) + " |"
    lines = [head, "|" + "---|" * (3 + len(METRICS))]
    for name, n, ep, cells in rows:
        lines.append(f"| {name} | {n} | {ep} | " + " | ".join(cells[k] for k, _, _ in METRICS) + " |")
    text = "\n".join(lines)
    print(text)
    with open(args.out, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
