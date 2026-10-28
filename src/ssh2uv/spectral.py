"""Isotropic kinetic-energy spectrum and spectral kinetic-energy flux on a uniform grid.

Physics
-------
For horizontal velocity u = (u, v), the kinetic energy per unit mass in wavenumber
k = (k_x, k_y) is E(k) = |u_hat(k)|^2 / 2. Its tendency due to horizontal advection is
the spectral transfer

    T(k) = -Re[ conj(u_hat(k)) . F{(u . grad) u}(k) ],

the rate at which nonlinear advection adds kinetic energy at k (m^2 s^-3 per unit
wavenumber). Advection only redistributes energy between scales, so for a periodic,
non-divergent flow T sums to zero over all k. The spectral flux through the isotropic
wavenumber K,

    Pi(K) = sum_{|k| >= K} T(k),

is the rate at which energy is transferred from wavenumbers below K to wavenumbers
above K. Pi > 0 is a forward (downscale) cascade toward small scales; Pi < 0 is an
inverse (upscale) cascade toward large scales, as found from altimetric geostrophic
velocities by Scott and Wang (2005, J. Phys. Oceanogr. 35, 1650-1666).

Numerics
--------
Velocity derivatives use second-order centered differences (one-sided at the edges).
Because the fields are not periodic, both the velocity and the advection term are
multiplied by the same 2-D Tukey window before the Fourier transform. With this window,
sum_k T(k) equals the windowed domain average of -u.(u.grad)u. That average is not
exactly zero: the flow is divergent and the edges are tapered. It is therefore reported
alongside Pi as `total_transfer`.
"""

from typing import Dict

import numpy as np


def tukey_2d(ny: int, nx: int, alpha: float = 0.25) -> np.ndarray:
    """Separable 2-D Tukey (tapered cosine) window."""
    def tukey(n):
        w = np.ones(n)
        m = int(np.floor(alpha * (n - 1) / 2))
        if m > 0:
            x = np.arange(m + 1)
            ramp = 0.5 * (1 - np.cos(np.pi * x / m))
            w[:m + 1] = ramp
            w[-(m + 1):] = ramp[::-1]
        return w
    return np.outer(tukey(ny), tukey(nx))


def advection(u: np.ndarray, v: np.ndarray, dx: float):
    """(u . grad) u and (u . grad) v for (..., ny, nx) fields; y along axis -2, x along axis -1."""
    du_dy, du_dx = np.gradient(u, dx, axis=(-2, -1))
    dv_dy, dv_dx = np.gradient(v, dx, axis=(-2, -1))
    return u * du_dx + v * du_dy, u * dv_dx + v * dv_dy


def radial_bins(ny: int, nx: int, dx: float):
    """Isotropic wavenumber of every Fourier mode and the bin edges (rad/m).

    The bin width is the fundamental wavenumber of the shorter side of the domain, so
    every bin contains modes.
    """
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dx)
    kmag = np.hypot(*np.meshgrid(kx, ky))
    dk = 2 * np.pi / (min(nx, ny) * dx)
    kmax = np.pi / dx  # Nyquist
    edges = np.arange(0.5 * dk, kmax + dk, dk)
    return kmag, edges, dk


def spectral_flux(u: np.ndarray, v: np.ndarray, dx: float, alpha: float = 0.25) -> Dict[str, np.ndarray]:
    """Time-mean isotropic KE spectrum and spectral KE flux of (nt, ny, nx) velocity fields.

    Returns a dict with
        k          bin-center wavenumbers (rad/m)
        wavelength 2 pi / k (m)
        E          kinetic-energy spectrum, m^3 s^-2 (energy per unit wavenumber)
        T          spectral transfer per unit wavenumber, m^3 s^-3
        Pi         spectral flux Pi(K) at the lower edge K of each bin, m^2 s^-3
        total_transfer  sum of T over all modes, m^2 s^-3 (zero for periodic non-divergent flow)
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.ndim == 2:
        u, v = u[None], v[None]
    nt, ny, nx = u.shape
    win = tukey_2d(ny, nx, alpha)
    adv_u, adv_v = advection(u, v, dx)
    n2 = float(ny * nx) ** 2
    uh = np.fft.fft2(u * win)
    vh = np.fft.fft2(v * win)
    au = np.fft.fft2(adv_u * win)
    av = np.fft.fft2(adv_v * win)
    # mean over time of the mode-wise energy and transfer (normalized so that sums over
    # modes are domain averages of the windowed fields)
    e_mode = (0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2) / n2).mean(axis=0)
    t_mode = (-np.real(np.conj(uh) * au + np.conj(vh) * av) / n2).mean(axis=0)

    kmag, edges, dk = radial_bins(ny, nx, dx)
    idx = np.digitize(kmag.ravel(), edges) - 1          # bin index; -1 for k < edges[0] (mean mode)
    nb = len(edges) - 1
    ok = (idx >= 0) & (idx < nb)
    e_bin = np.bincount(idx[ok], weights=e_mode.ravel()[ok], minlength=nb)
    t_bin = np.bincount(idx[ok], weights=t_mode.ravel()[ok], minlength=nb)
    # flux through the lower edge of each bin: transfer summed over all modes with |k| >= K
    pi = np.cumsum(t_bin[::-1])[::-1]
    k = 0.5 * (edges[:-1] + edges[1:])
    return {
        "k": k, "wavelength": 2 * np.pi / k, "k_lower": edges[:-1],
        "E": e_bin / dk, "T": t_bin / dk, "Pi": pi,
        "total_transfer": np.array(t_mode.sum()),
    }
