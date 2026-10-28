#!/usr/bin/env python3
"""Kinetic-energy spectra and spectral KE fluxes (paper Figure 7, Section 6).

Top row: isotropic KE spectrum E(K); bottom row: spectral flux Pi(K) (Pi > 0: forward
cascade to small scales, Pi < 0: inverse cascade to large scales). One column per input
file written by scripts/spectral_flux.py (e.g. 53 m and 0.5 m). Shading: one standard
deviation over seeds. Wavelengths above --max_km (the domain scale) are not shown because
open boundaries and the window affect them.

Usage:
    python scripts/figures/fig_spectral_flux.py \
        --inputs results/spectral_flux/spectral_flux_53m.npz results/spectral_flux/spectral_flux_surface.npz \
        --titles "53 m" "0.5 m (surface)" --out figures/spectral_flux
"""

import argparse

import numpy as np

from ssh2uv.plotting import TEXT_WIDTH_IN, panel_label, paper_style, plt, save

STYLE = {  # key: (label, color, linestyle, linewidth)
    "truth": ("LLC4320 (truth)", "#0b0b0b", "-", 1.8),
    "geostrophic": ("Geostrophic", "#8a8985", "--", 1.5),
    "e0": ("Baseline U-Net", "#eb6834", "-", 1.3),
    "c3": ("Final U-Net", "#2a78d6", "-", 1.6),
}


def load(path):
    z = np.load(path)
    keys = sorted({k.split("__")[0] for k in z.files})
    return {key: {f.split("__")[1]: z[f] for f in z.files if f.startswith(key + "__")} for key in keys}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", default=["results/spectral_flux/spectral_flux_53m.npz",
                                                 "results/spectral_flux/spectral_flux_surface.npz"])
    p.add_argument("--titles", nargs="+", default=["53 m", "0.5 m (surface)"])
    p.add_argument("--max_km", type=float, default=128.0)
    p.add_argument("--out", default="figures/spectral_flux")
    a = p.parse_args()
    paper_style()
    ncol = len(a.inputs)
    fig, axes = plt.subplots(2, ncol, figsize=(TEXT_WIDTH_IN if ncol > 1 else 0.62 * TEXT_WIDTH_IN, 4.6),
                             sharex=True, squeeze=False)
    fig.subplots_adjust(left=0.13 if ncol > 1 else 0.2, right=0.98, top=0.93, bottom=0.11, hspace=0.12, wspace=0.28)
    letters = iter("abcdefgh")
    for j, path in enumerate(a.inputs):
        data = load(path)
        lam = data["truth"]["wavelength"] / 1e3
        keep = lam <= a.max_km
        ax_e, ax_p = axes[0, j], axes[1, j]
        for key in ("truth", "geostrophic", "e0", "c3"):
            if key not in data:
                continue
            label, color, ls, lw = STYLE[key]
            d = data[key]
            ax_e.loglog(lam[keep], d["E"][keep], color=color, ls=ls, lw=lw, label=label)
            ax_p.semilogx(lam[keep], d["Pi"][keep] * 1e9, color=color, ls=ls, lw=lw, label=label)
            if "Pi_seed_std" in d and np.any(d["Pi_seed_std"]):
                ax_p.fill_between(lam[keep], (d["Pi"] - d["Pi_seed_std"])[keep] * 1e9,
                                  (d["Pi"] + d["Pi_seed_std"])[keep] * 1e9, color=color, alpha=0.2, lw=0)
        ax_p.axhline(0, color="#52514e", lw=0.8)
        for ax in (ax_e, ax_p):
            ax.set_xlim(a.max_km, lam.min())       # large scales on the left
            ax.grid(color="#e6e5e1", lw=0.6, which="major")
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        if a.titles:
            ax_e.set_title(a.titles[j])
        ax_p.set_xlabel("wavelength (km)")
        if j == 0:
            ax_e.set_ylabel(r"$E(K)$ (m$^3$ s$^{-2}$)")
            ax_p.set_ylabel(r"$\Pi(K)$ ($10^{-9}$ m$^2$ s$^{-3}$)")
        panel_label(ax_e, next(letters))
        panel_label(ax_p, next(letters))
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.set_xticks([100, 50, 20, 10, 5])
        ax.set_xticklabels(["100", "50", "20", "10", "5"])
    save(fig, a.out)


if __name__ == "__main__":
    main()
