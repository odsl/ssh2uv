#!/usr/bin/env python3
"""Snapshot of the model fields used in this study (paper Figure 1, Section 3.1).

Panels (one time step, the six 64x64 patches stitched into the 256 km x 384 km domain):
    (a) sea surface height anomaly eta
    (b) sea surface temperature
    (c) wind-stress vectors over |tau|
    (d) 53 m velocity vectors over speed
SSH contours are drawn on every panel.

Usage:
    python scripts/figures/fig_snapshot.py --data_dir data --time_index 5000 --out figures/snapshot
"""

import argparse
import datetime as dt
import os

import numpy as np

from ssh2uv.kinematics import DOMAIN_LON, DX_M, N_PATCH_COLS, N_PATCH_ROWS, N_PATCHES, PATCH, \
    lat_to_y_km, stitch_patches, y_km_to_lat
from ssh2uv.plotting import TEXT_WIDTH_IN, bottom_colorbar, panel_label, paper_style, plt, save

# Time of sample index t: first granule starts 2011-09-13 00:00 and preprocessing
# trims the first 50 hours (EDGE_TRIM in scripts/preprocess.py).
T0 = dt.datetime(2011, 9, 13, 0, 0)
EDGE_TRIM_H = 50


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=os.environ.get("SSH2UV_DATA", "data"))
    p.add_argument("--time_index", type=int, default=5000)
    p.add_argument("--out", default="figures/snapshot")
    p.add_argument("--stride", type=int, default=12, help="Vector thinning (grid points)")
    return p.parse_args()


def quiver_key(fig, ax, q, value, label):
    """Reference vector on a white box in the lower-right corner of *ax*."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((0.33, 0.015), 0.655, 0.085, boxstyle="round,pad=0.005",
                                transform=ax.transAxes, fc="white", ec="0.5", lw=0.5,
                                alpha=0.9, zorder=4))
    key = ax.quiverkey(q, 0.40, 0.055, value, label, labelpos="E", coordinates="axes",
                       labelsep=0.04, fontproperties={"size": 8.5})
    key.set_zorder(5)


def main():
    args = parse_args()
    paper_style()
    raw = np.load(os.path.join(args.data_dir, "global-final-64_stacked_DEEP.npy"), mmap_mode="r")
    t = args.time_index
    fields = stitch_patches(np.asarray(raw[t * N_PATCHES:(t + 1) * N_PATCHES]))  # (6, 192, 128)
    eta, taux, tauy, sst, u, v = fields
    when = T0 + dt.timedelta(hours=EDGE_TRIM_H + t)
    print(f"time index {t} -> {when:%Y-%m-%d %H:%M} UTC")

    ny, nx = eta.shape
    dx_km = DX_M / 1000.0
    x = (np.arange(nx) + 0.5) * dx_km
    y = (np.arange(ny) + 0.5) * dx_km
    extent = [0, nx * dx_km, 0, ny * dx_km]
    X, Y = np.meshgrid(x, y)

    eta_c = eta - eta.mean()
    span = np.percentile(eta_c, 99.5) - np.percentile(eta_c, 0.5)
    step = 0.02 if span < 0.4 else 0.05
    levels = np.arange(np.floor(eta_c.min() / step) * step, eta_c.max() + step, step)

    fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH_IN, 3.9), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.9, top=0.88, bottom=0.30, wspace=0.12)

    def contours(ax, color="k", alpha=0.55):
        ax.contour(X, Y, eta_c, levels=levels, colors=color, linewidths=0.4, alpha=alpha)

    def imshow(ax, data, **kw):
        return ax.imshow(data, origin="lower", extent=extent, interpolation="nearest", **kw)

    # (a) SSH
    lim = np.percentile(np.abs(eta_c), 99.5)
    im = imshow(axes[0], eta_c, cmap="RdBu_r", vmin=-lim, vmax=lim)
    contours(axes[0])
    for r in range(1, N_PATCH_ROWS):
        axes[0].axhline(r * PATCH * dx_km, color="0.15", lw=0.9, ls="--")
    for c in range(1, N_PATCH_COLS):
        axes[0].axvline(c * PATCH * dx_km, color="0.15", lw=0.9, ls="--")
    bottom_colorbar(fig, im, axes[0], r"$\eta - \langle\eta\rangle$ (m)", pad=0.135)

    # (b) SST
    lo, hi = np.percentile(sst, [0.5, 99.5])
    im = imshow(axes[1], sst, cmap="inferno", vmin=lo, vmax=hi)
    contours(axes[1], color="w", alpha=0.6)
    bottom_colorbar(fig, im, axes[1], r"SST $\theta$ ($^\circ$C)", pad=0.135)

    # (c) wind stress
    s = args.stride
    sl = (slice(s // 2, None, s), slice(s // 2, None, s))
    sw = int(round(s * 4 / 3))  # wind varies on larger scales; fewer vectors
    slw = (slice(sw // 2, None, sw), slice(sw // 2, None, sw))
    tau = np.hypot(taux, tauy)
    im = imshow(axes[2], tau, cmap="Blues", vmin=0, vmax=np.percentile(tau, 99.5))
    contours(axes[2])
    q = axes[2].quiver(X[slw], Y[slw], taux[slw], tauy[slw], color="k", scale_units="width",
                       scale=float(np.percentile(tau, 95)) * 9, width=0.012, headwidth=3.5)
    ref_tau = float(np.round(np.percentile(tau, 90), 2)) or 0.1
    quiver_key(fig, axes[2], q, ref_tau, f"{ref_tau:g} N m$^{{-2}}$")
    bottom_colorbar(fig, im, axes[2], r"$|\boldsymbol{\tau}|$ (N m$^{-2}$)", pad=0.135)

    # (d) velocity at 53 m
    spd = np.hypot(u, v)
    im = imshow(axes[3], spd, cmap="YlOrRd", vmin=0, vmax=np.percentile(spd, 99.5))
    contours(axes[3])
    q = axes[3].quiver(X[sl], Y[sl], u[sl], v[sl], color="k", scale_units="width",
                       scale=float(np.percentile(spd, 95)) * 6, width=0.012, headwidth=3.5)
    ref_u = float(np.round(np.percentile(spd, 90), 1)) or 0.2
    quiver_key(fig, axes[3], q, ref_u, f"{ref_u:g} m s$^{{-1}}$")
    bottom_colorbar(fig, im, axes[3], r"$|\mathbf{u}|$ at 53 m (m s$^{-1}$)", pad=0.135)

    for ax, lab in zip(axes, "abcd"):
        panel_label(ax, lab)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks([0, 100, 200])
        ax.set_xlabel("x (km)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y (km)")
    lat_ax = axes[3].secondary_yaxis("right", functions=(y_km_to_lat, lat_to_y_km))
    lat_ticks = np.arange(-44.5, -41.4, 1.0)
    lat_ax.set_yticks(lat_ticks)
    lat_ax.set_yticklabels([f"{abs(t):.1f}°S" for t in lat_ticks])

    print(f"SSH contour interval {step} m; domain lon {DOMAIN_LON[0]}°E at x=0; "
          f"lat {y_km_to_lat(0):.2f} to {y_km_to_lat(extent[3]):.2f}")
    save(fig, args.out)


if __name__ == "__main__":
    main()
