"""Shared figure style and helpers for the paper figures.

Figures are drawn at their printed size (the preprint text block is 6.5 in wide)
so that font sizes in points are the sizes that appear on the page; nothing is
shrunk when the PNG/PDF is placed with ``\\includegraphics[width=\\textwidth]``.
"""

import os
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TEXT_WIDTH_IN = 6.5

PAPER_RC = {
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "mathtext.fontset": "dejavusans",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "figure.facecolor": "white",
}

PANEL_LABEL_SIZE = 10.5


def paper_style() -> None:
    """Apply the paper rcParams globally."""
    plt.rcParams.update(PAPER_RC)


def panel_label(ax, label: str, loc: str = "upper left", inside: bool = True) -> None:
    """Bold '(a)'-style label in a white box at a corner of *ax*."""
    x = 0.03 if "left" in loc else 0.97
    y = 0.97 if "upper" in loc else 0.03
    ax.text(
        x, y, f"({label})", transform=ax.transAxes,
        ha="left" if "left" in loc else "right",
        va="top" if "upper" in loc else "bottom",
        fontsize=PANEL_LABEL_SIZE, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8) if inside else None,
    )


def row_colorbar(fig, mappable, axes_row: Iterable, label: str, pad: float = 0.012,
                 width: float = 0.012, extend: str = "neither", ticks=None):
    """Colorbar in its own axes to the right of a row of panels (never over data)."""
    axes_row = list(axes_row)
    fig.canvas.draw_idle()
    boxes = [ax.get_position() for ax in axes_row]
    x1 = max(b.x1 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    cax = fig.add_axes([x1 + pad, y0, width, y1 - y0])
    cb = fig.colorbar(mappable, cax=cax, extend=extend, ticks=ticks)
    cb.set_label(label)
    cb.outline.set_linewidth(0.6)
    return cb


def bottom_colorbar(fig, mappable, ax, label: str, pad: float = 0.012, height: float = 0.014,
                    extend: str = "neither", ticks=None):
    """Horizontal colorbar in its own axes directly below *ax*."""
    fig.canvas.draw_idle()
    b = ax.get_position()
    cax = fig.add_axes([b.x0, b.y0 - pad - height, b.width, height])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend=extend, ticks=ticks)
    cb.set_label(label)
    cb.outline.set_linewidth(0.6)
    return cb


def scale_bar(ax, length_km: float, dx_km: float, label: Optional[str] = None,
              loc=(0.06, 0.07), color="k") -> None:
    """Horizontal scale bar on an image axis whose pixels are *dx_km* wide."""
    x0_frac, y0_frac = loc
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + x0_frac * (xmax - xmin)
    y0 = ymin + y0_frac * (ymax - ymin)
    n_px = length_km / dx_km
    ax.plot([x0, x0 + n_px], [y0, y0], color=color, lw=2.2, solid_capstyle="butt")
    ax.text(x0 + n_px / 2, y0 + 0.025 * (ymax - ymin), label or f"{length_km:g} km",
            ha="center", va="bottom", color=color, fontsize=9)


def symmetric_limit(field: np.ndarray, pct: float = 98.0) -> float:
    """Symmetric colour limit from a percentile of |field| (NaNs ignored)."""
    return float(np.nanpercentile(np.abs(field), pct))


def save(fig, out_base: str, formats=("png", "pdf")) -> None:
    """Save *fig* as ``out_base.png`` and ``out_base.pdf``."""
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    for ext in formats:
        fig.savefig(f"{out_base}.{ext}")
        print(f"Saved {out_base}.{ext}")
