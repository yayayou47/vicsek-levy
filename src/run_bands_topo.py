"""
Travelling-band detection for the topological-Vicsek variant. Same
projection and band-index protocol as run_bands.py, but using
TopoVicsek (k = 6 nearest-neighbour alignment, metric repulsion).
We probe the near-transition regime located by the topological FSS
(eta around 0.20-0.30 for alpha = 1, eta around 0.45-0.50 for
alpha = 2).

Output: data/bands_topo.npz
   alphas, etas, centers, profiles[ia, k] (time-averaged)
   band_idx[ia], snap_x[ia, n], snap_y[ia, n], snap_theta[ia, n]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from topological import TopoVicsek, TopoParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def project_and_hist(x, y, theta_avg, L, n_bins):
    cx, sx = np.cos(theta_avg), np.sin(theta_avg)
    x_par = (x * cx + y * sx) % L
    counts, edges = np.histogram(x_par, bins=n_bins, range=(0.0, L))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def main():
    N, L, v0 = 2000, 30.0, 0.05
    R_r = 0.45
    k_topo = 6
    n_warm = 2000
    n_meas = 4000
    n_skip = 25
    n_bins = 60
    seeds = [11, 23, 41]

    # Near-transition operating points for the topological variant,
    # one per alpha. Calibrated from data/topo_fss.npz: at L = 30,
    # alpha = 1 has its U_4 dip around eta = 0.30, alpha = 2 around
    # eta = 0.50.
    cases = [(2.0, 0.40), (1.5, 0.30), (1.0, 0.20)]

    profiles = np.zeros((len(cases), n_bins))
    band_idx = np.zeros(len(cases))
    snap_x = np.zeros((len(cases), N))
    snap_y = np.zeros((len(cases), N))
    snap_theta = np.zeros((len(cases), N))
    centers = None

    pbar = tqdm(total=len(cases) * len(seeds), desc="bands_topo")
    for ic, (alpha, eta) in enumerate(cases):
        sum_prof = np.zeros(n_bins)
        n_acc = 0
        for is_, seed in enumerate(seeds):
            p = TopoParams(
                N=N, L=L, v0=v0, R_r=R_r, k=k_topo,
                eta=float(eta),
                alpha=float(alpha), seed=seed,
            )
            sim = TopoVicsek(p)
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
                    if centers is None:
                        centers = cs
                    sum_prof += cn
                    n_acc += 1
            if is_ == 0:
                snap_x[ic] = sim.x
                snap_y[ic] = sim.y
                snap_theta[ic] = sim.theta
            pbar.update(1)
        prof = sum_prof / n_acc
        profiles[ic] = prof
        band_idx[ic] = prof.max() / prof.mean() if prof.mean() > 0 else 1.0
    pbar.close()

    alphas = np.array([c[0] for c in cases])
    etas = np.array([c[1] for c in cases])

    np.savez_compressed(
        DATA / "bands_topo.npz",
        alphas=alphas, etas=etas, centers=centers,
        profiles=profiles, band_idx=band_idx,
        snap_x=snap_x, snap_y=snap_y, snap_theta=snap_theta,
        params=np.array([N, L, v0, R_r, k_topo, n_warm, n_meas,
                         n_skip, len(seeds)], dtype=float),
    )
    print("saved:", DATA / "bands_topo.npz")
    print("band indices:", band_idx)


if __name__ == "__main__":
    main()
