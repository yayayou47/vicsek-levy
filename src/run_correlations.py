"""
Spatial correlation functions in the zonal Vicsek-Couzin model.

   C_v(r) = <cos(theta_i - theta_j)>_{|r_ij| = r}    velocity correlation
   g(r)   = pair-density / ideal-gas density          pair correlation

For each alpha the noise scale is swept across the transition;
C_v(r) is fit to an exponential decay to extract the correlation
length xi(eta), which should diverge near eta_c.

Output: data/correlations.npz with arrays
   alphas, etas, centers, Cv[a, e, k], g[a, e, k], xi[a, e]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def pair_histograms(x, y, theta, L, bins, r_max):
    """Return (counts, sum_cos) histograms over pair distances.
    counts[k] is the number of (i, j != i) pairs with d in bin k;
    sum_cos[k] is the sum of cos(theta_i - theta_j) over those pairs.
    """
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    dx -= L * np.round(dx / L)
    dy -= L * np.round(dy / L)
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    cos_diff = np.cos(theta[None, :] - theta[:, None])
    mask = d < r_max
    counts, _ = np.histogram(d[mask], bins=bins)
    sum_c, _ = np.histogram(d[mask], bins=bins, weights=cos_diff[mask])
    return counts, sum_c


def main():
    N, L, v0 = 500, 15.0, 0.05
    R_r, R_a = 0.5, 0.7
    n_warm, n_meas, n_skip = 1500, 1000, 50
    n_bins = 40
    r_max = 7.0
    seeds = [11, 23, 41]

    alphas = np.array([1.0, 1.5, 2.0])
    etas = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])

    bins = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    dr = bins[1] - bins[0]

    n_a, n_e = len(alphas), len(etas)
    counts = np.zeros((n_a, n_e, n_bins))
    sumc = np.zeros((n_a, n_e, n_bins))

    pbar = tqdm(total=n_a * n_e * len(seeds), desc="corr")
    for ia, alpha in enumerate(alphas):
        for ie, eta in enumerate(etas):
            for seed in seeds:
                p = VicsekParams(
                    N=N, L=L, v0=v0, R_r=R_r, R_a=R_a,
                    eta=float(eta), alpha=float(alpha), seed=seed,
                )
                sim = Vicsek(p)
                sim.theta[:] = 0.0
                for _ in range(n_warm):
                    sim.step()
                for k in range(n_meas):
                    sim.step()
                    if k % n_skip == 0:
                        cnt, sc = pair_histograms(
                            sim.x, sim.y, sim.theta, L, bins, r_max
                        )
                        counts[ia, ie] += cnt
                        sumc[ia, ie] += sc
                pbar.update(1)
    pbar.close()

    Cv = np.where(counts > 0, sumc / counts, 0.0)
    n_snap = (n_meas // n_skip) * len(seeds)
    annulus_area = 2.0 * np.pi * centers * dr
    g = counts * L**2 / (N**2 * annulus_area * n_snap)

    # Correlation length xi from log(Cv) ~ -r/xi over r in [1, 5],
    # capped at L/2 to avoid fits dominated by saturation at eta = 0.
    fit_mask = (centers >= 1.0) & (centers <= 5.0)
    xi = np.full((n_a, n_e), np.nan)
    for ia in range(n_a):
        for ie in range(n_e):
            cv = Cv[ia, ie][fit_mask]
            rs = centers[fit_mask]
            valid = cv > 0.02
            if valid.sum() >= 3:
                slope, _ = np.polyfit(rs[valid], np.log(cv[valid]), 1)
                if slope < 0:
                    xi[ia, ie] = min(-1.0 / slope, L / 2.0)

    np.savez_compressed(
        DATA / "correlations.npz",
        alphas=alphas, etas=etas, centers=centers,
        Cv=Cv, g=g, xi=xi,
        params=np.array([N, L, v0, R_r, R_a, n_warm, n_meas, n_skip,
                         n_bins, r_max, len(seeds)], dtype=float),
    )
    print("saved:", DATA / "correlations.npz")


if __name__ == "__main__":
    main()
