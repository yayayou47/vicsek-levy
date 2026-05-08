"""
Observables for the Vicsek phase transition.

Order parameter           : phi  = | <e^{i theta_i}> |
Susceptibility            : chi  = N * Var(phi)
Binder cumulant           : U4   = 1 - <phi^4> / (3 <phi^2>^2)
Density giant fluctuations: scaling Var(N_box) vs <N_box> in subboxes.
"""
from __future__ import annotations

import numpy as np


def time_series_stats(phi: np.ndarray, N: int) -> dict:
    """phi: 1D array of polarisation samples after equilibration."""
    m1 = float(np.mean(phi))
    m2 = float(np.mean(phi**2))
    m4 = float(np.mean(phi**4))
    chi = N * (m2 - m1**2)
    U4 = 1.0 - m4 / (3.0 * m2**2 + 1e-30)
    return dict(phi_mean=m1, phi_var=m2 - m1**2, chi=chi, binder=U4)


def density_fluctuations(
    x: np.ndarray, y: np.ndarray, L: float, n_box_sizes: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_N, var_N) for nested sub-boxes -- giant number fluctuations test."""
    sizes = np.geomspace(L / 32.0, L / 2.0, n_box_sizes)
    means = np.empty_like(sizes)
    variances = np.empty_like(sizes)
    for k, s in enumerate(sizes):
        n_grid = max(2, int(L / s))
        H, _, _ = np.histogram2d(
            x, y, bins=n_grid, range=[[0.0, L], [0.0, L]]
        )
        means[k] = H.mean()
        variances[k] = H.var()
    return means, variances
