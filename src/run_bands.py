"""
Travelling-band detection. For each alpha at near-critical noise, we
run a long simulation, project particle positions on the global polar
direction, and time-average the resulting density profile. A
pronounced peak in the profile indicates a band soliton; a flat
profile indicates a homogeneous polar fluid.

Quantitative metric: band index = max(sigma)/<sigma> over the
projection axis.

Output: data/bands.npz
   alphas, etas, centers, profiles[ia, k]  (time-averaged)
   band_idx[ia], snap_x[ia, n], snap_y[ia, n], snap_theta[ia, n]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def project_and_hist(x, y, theta_avg, L, n_bins):
    """Project positions onto the global polar-order axis (mod L) and
    return centers, counts."""
    cx, sx = np.cos(theta_avg), np.sin(theta_avg)
    x_par = (x * cx + y * sx) % L
    counts, edges = np.histogram(x_par, bins=n_bins, range=(0.0, L))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def main():
    N, L, v0 = 2000, 30.0, 0.05
    R_r, R_a = 0.5, 0.7
    n_warm = 2500
    n_meas = 5000
    n_skip = 25
    n_bins = 60
    seeds = [11, 23, 41]

    # Near-critical (ordered side) for each alpha: eta just below the
    # FSS-located eta_c.
    cases = [(2.0, 0.13), (1.5, 0.08), (1.0, 0.04)]

    profiles = np.zeros((len(cases), n_bins))
    band_idx = np.zeros(len(cases))
    snap_x = np.zeros((len(cases), N))
    snap_y = np.zeros((len(cases), N))
    snap_theta = np.zeros((len(cases), N))
    centers = None

    pbar = tqdm(total=len(cases) * len(seeds), desc="bands")
    for ic, (alpha, eta) in enumerate(cases):
        sum_prof = np.zeros(n_bins)
        n_acc = 0
        for is_, seed in enumerate(seeds):
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
                    cx = np.cos(sim.theta).sum()
                    sxs = np.sin(sim.theta).sum()
                    theta_avg = float(np.arctan2(sxs, cx))
                    cs, cn = project_and_hist(sim.x, sim.y, theta_avg,
                                              L, n_bins)
                    sum_prof += cn
                    n_acc += 1
                    centers = cs
            if is_ == 0:
                snap_x[ic] = sim.x.copy()
                snap_y[ic] = sim.y.copy()
                snap_theta[ic] = sim.theta.copy()
            pbar.update(1)
        profiles[ic] = sum_prof / n_acc
        band_idx[ic] = float(profiles[ic].max()
                             / profiles[ic].mean())
    pbar.close()

    np.savez_compressed(
        DATA / "bands.npz",
        alphas=np.array([c[0] for c in cases]),
        etas=np.array([c[1] for c in cases]),
        profiles=profiles, centers=centers,
        band_idx=band_idx,
        snap_x=snap_x, snap_y=snap_y, snap_theta=snap_theta,
        params=np.array([N, L, v0, R_r, R_a, n_warm, n_meas,
                         n_skip, n_bins, len(seeds)], dtype=float),
    )
    print("saved:", DATA / "bands.npz")


if __name__ == "__main__":
    main()
