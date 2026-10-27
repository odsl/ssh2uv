#!/usr/bin/env python3
"""Paper-style ablation tables for the runs around the final configuration (C3).

Usage:
    python scripts/summarize_ablation.py --runs results/temporal --out results/ablation.md
"""

import argparse
import glob
import json
import os

import numpy as np

REF = "C3_bc128_phys_long"  # gradients + SST + wind, width 128, MSE + physics penalties


def load(runs, name):
    files = sorted(glob.glob(os.path.join(runs, name, "seed*", "results.json")))
    return [json.load(open(f)) for f in files]


def stat(res, split, key, d):
    v = np.array([r[split][key] for r in res])
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}" if len(v) > 1 else (f"{v[0]:.{d}f}" if len(v) else "–")


def row(res, label):
    if not res:
        return f"| {label} | – | – | – | – | – | – |"
    return (f"| {label} | {len(res)} | {stat(res, 'test', 'rmse_speed', 4)} | {stat(res, 'test', 'mean_dir_error_deg', 1)} | "
            f"{stat(res, 'test', 'r_zeta', 3)} | {stat(res, 'test', 'r_div', 3)} | {stat(res, 'val', 'rmse_speed', 4)} |")


HEAD = ("| model | seeds | test RMSE speed (m/s) | test dir. err (°) | test r(ζ) | test r(δ) | val RMSE speed (m/s) |\n"
        "|---|---|---|---|---|---|---|")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/temporal", help="results/temporal (shipped) or runs/temporal")
    p.add_argument("--geostrophic", default="results/temporal_retrain_metrics.json")
    p.add_argument("--out", default="results/ablation.md")
    a = p.parse_args()
    L = lambda n: load(a.runs, n)
    out = []
    if os.path.exists(a.geostrophic):
        g = json.load(open(a.geostrophic))["geostrophic"]
        out.append(f"Geostrophic reference (no training): test RMSE speed {g['test']['rmse_speed']:.4f} m/s, "
                   f"dir. err {g['test']['mean_dir_error_deg']:.1f}°, r(ζ) {g['test']['r_zeta']:.3f}, r(δ) 0; "
                   f"val RMSE speed {g['val']['rmse_speed']:.4f} m/s.\n")

    out.append("## 1. SSH representation × covariates (width 128, MSE + physics penalties)\n")
    out.append("| covariates | raw SSH: test RMSE speed | SSH gradients: test RMSE speed | change | raw SSH: test r(ζ) / r(δ) | gradients: test r(ζ) / r(δ) |")
    out.append("|---|---|---|---|---|---|")
    for cov, en, gn in [("none", "A_eta", "A_grad"), ("SST", "A_eta_sst", "A_grad_sst"),
                        ("wind", "A_eta_wind", "A_grad_wind"), ("SST + wind", "A_eta_sst_wind", REF)]:
        e, g = L(en), L(gn)
        if e and g:
            me, mg = np.mean([r["test"]["rmse_speed"] for r in e]), np.mean([r["test"]["rmse_speed"] for r in g])
            ch = f"{100 * (mg / me - 1):+.1f} %"
        else:
            ch = "–"
        rz = lambda res: (f"{np.mean([r['test']['r_zeta'] for r in res]):.3f} / {np.mean([r['test']['r_div'] for r in res]):.3f}"
                          if res else "–")
        out.append(f"| {cov} | {stat(e, 'test', 'rmse_speed', 4)} | {stat(g, 'test', 'rmse_speed', 4)} | {ch} | {rz(e)} | {rz(g)} |")
    out.append("\nSame runs with validation scores:\n")
    out.append(HEAD)
    for n, lab in [("A_eta", "raw SSH"), ("A_eta_sst", "raw SSH + SST"), ("A_eta_wind", "raw SSH + wind"),
                   ("A_eta_sst_wind", "raw SSH + SST + wind"), ("A_grad", "gradients"), ("A_grad_sst", "gradients + SST"),
                   ("A_grad_wind", "gradients + wind"), (REF, "gradients + SST + wind (C3)")]:
        out.append(row(L(n), lab))

    out.append("\n## 2. Capacity (gradients + SST + wind, MSE + physics penalties)\n")
    out.append(HEAD)
    for n, lab in [("A_bc64", "width 64 (31M)"), (REF, "width 128 (124M, C3)"), ("A_bc192", "width 192 (279M)")]:
        out.append(row(L(n), lab))

    out.append("\n## 3. Training objective (gradients + SST + wind, width 128)\n")
    out.append(HEAD)
    for n, lab in [("A_mse", "MSE"), ("A_mae", "MAE"), ("A_mae_phys", "MAE + vorticity/divergence penalties"),
                   (REF, "MSE + vorticity/divergence penalties (C3)")]:
        out.append(row(L(n), lab))

    text = "\n".join(out)
    print(text)
    with open(a.out, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
