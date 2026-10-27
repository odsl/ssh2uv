#!/usr/bin/env python3
"""Joint PDF of vorticity and strain (paper Figure 7), after Balwada et al. (2021).

x: zeta/f with signed f (zeta/f > 0 is cyclonic; f < 0 here)
y: sigma/|f|, sigma = sqrt((u_x - v_y)^2 + (u_y + v_x)^2) >= 0
Left: LLC4320 truth; right: U-Net prediction, over all pixels of the given block.
The PDF is normalised by the total number of pixels, so pixels outside the
plotted range still count; that fraction is printed and saved.
Reads the arrays written by scripts/predict.py.

Usage:
    python scripts/figures/fig_vorticity_strain_pdf.py --eval_dir outputs/test_predictions --model c3
"""

import argparse
import json
import os

import numpy as np
import torch
from matplotlib.colors import LogNorm

from ssh2uv.evaluation import PAPER_MODELS_BY_KEY
from ssh2uv.kinematics import F0, strain, vorticity
from ssh2uv.plotting import TEXT_WIDTH_IN, paper_style, plt, row_colorbar, save


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_dir", default="outputs/test_predictions")
    p.add_argument("--model", default="c3", help="prediction key in <eval_dir>/predictions")
    p.add_argument("--label", default="final model", help="Panel title for the model")
    p.add_argument("--nbins", type=int, default=160)
    p.add_argument("--out", default="figures/vorticity_strain_pdf")
    return p.parse_args()


def zeta_sigma(uv, chunk=1024):
    z, s = [], []
    for i in range(0, len(uv), chunk):
        t = torch.from_numpy(np.ascontiguousarray(uv[i:i + chunk])).double()
        z.append((vorticity(t[:, 0], t[:, 1]) / F0).float().numpy().ravel())
        s.append((strain(t[:, 0], t[:, 1]) / abs(F0)).float().numpy().ravel())
    return np.concatenate(z), np.concatenate(s)


def joint_pdf(z, s, z_edges, s_edges):
    h, _, _ = np.histogram2d(z, s, bins=[z_edges, s_edges])
    area = (z_edges[1] - z_edges[0]) * (s_edges[1] - s_edges[0])
    pdf = h.T / (len(z) * area)
    inside = h.sum() / len(z)
    return np.ma.masked_where(pdf == 0, pdf), inside


def main():
    args = parse_args()
    paper_style()
    truth = np.load(os.path.join(args.eval_dir, "truth.npy"), mmap_mode="r")
    pred = np.load(os.path.join(args.eval_dir, "predictions", f"{args.model}.npy"), mmap_mode="r")
    zt, st = zeta_sigma(truth)
    zp, sp = zeta_sigma(pred)

    zmax = np.ceil(np.percentile(np.abs(zt), 99.9) / 0.25) * 0.25
    smax = np.ceil(np.percentile(st, 99.9) / 0.25) * 0.25
    z_edges = np.linspace(-zmax, zmax, args.nbins + 1)
    s_edges = np.linspace(0.0, smax, args.nbins // 2 + 1)
    pdf_t, in_t = joint_pdf(zt, st, z_edges, s_edges)
    pdf_p, in_p = joint_pdf(zp, sp, z_edges, s_edges)

    stats = {
        "n_pixels": int(len(zt)),
        "zeta_over_f_range": float(zmax), "sigma_over_absf_max": float(smax),
        "fraction_inside_truth": float(in_t), "fraction_inside_pred": float(in_p),
        "skewness_zeta_truth": float(((zt - zt.mean()) ** 3).mean() / zt.std() ** 3),
        "skewness_zeta_pred": float(((zp - zp.mean()) ** 3).mean() / zp.std() ** 3),
        "fraction_zeta_over_f_gt_0.5_truth": float((zt > 0.5).mean()),
        "fraction_zeta_over_f_lt_-0.5_truth": float((zt < -0.5).mean()),
        "fraction_sigma_gt_0.5_truth": float((st > 0.5).mean()),
        "fraction_sigma_gt_0.5_pred": float((sp > 0.5).mean()),
        "max_abs_zeta_over_f_truth": float(np.abs(zt).max()),
        "std_zeta_over_f": {"truth": float(zt.std()), "pred": float(zp.std())},
        "std_sigma_over_absf": {"truth": float(st.std()), "pred": float(sp.std())},
    }
    print(json.dumps(stats, indent=2))

    vmax = 10 ** np.ceil(np.log10(max(pdf_t.max(), pdf_p.max())))
    norm = LogNorm(vmin=vmax * 1e-5, vmax=vmax)
    z_c = 0.5 * (z_edges[1:] + z_edges[:-1])
    s_c = 0.5 * (s_edges[1:] + s_edges[:-1])

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.75), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.86, top=0.9, bottom=0.17, wspace=0.08)
    label = args.label or PAPER_MODELS_BY_KEY[args.model].label
    for ax, pdf, title in zip(axes, [pdf_t, pdf_p], ["(a) LLC4320 (truth)", f"(b) U-Net ({label})"]):
        im = ax.pcolormesh(z_c, s_c, pdf, norm=norm, cmap="magma_r", rasterized=True, shading="auto")
        ax.contour(z_c, s_c, pdf.filled(0), levels=[1e-3, 1e-2, 1e-1, 1.0], colors="w",
                   linewidths=0.6, alpha=0.8)
        ref = np.linspace(0, min(zmax, smax), 50)
        ax.plot(ref, ref, color="0.4", lw=0.8, ls=":")
        ax.plot(-ref, ref, color="0.4", lw=0.8, ls=":")
        ax.axvline(0, color="0.4", lw=0.7, ls="--")
        ax.set_xlim(-zmax, zmax)
        ax.set_ylim(0, smax)
        ax.set_xlabel(r"$\zeta/f$")
        ax.set_title(title, loc="left", fontweight="bold")
    axes[0].set_ylabel(r"$\sigma/|f|$")
    axes[0].text(0.97, 0.95, "cyclonic →", transform=axes[0].transAxes, ha="right", va="top", fontsize=8.5)
    axes[0].text(0.03, 0.95, "← anticyclonic", transform=axes[0].transAxes, ha="left", va="top",
                 fontsize=8.5)
    row_colorbar(fig, im, axes, "joint PDF")

    save(fig, args.out)
    with open(args.out + "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
