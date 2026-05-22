"""
Cluster-size distribution. Connected components in the contact graph
(edges for d_ij < R_a, periodic). For each (alpha, eta) we histogram
cluster sizes over many snapshots and also record the largest-cluster
fraction <s_max>/N as a coarse percolation order parameter.

Output: data/clusters.npz
   alphas, etas, hist_bins[k], hist[a, e, k], smax_frac[a, e]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def cluster_sizes(x, y, L, R):
    pts = np.column_stack([x, y]) % L
    tree = cKDTree(pts, boxsize=L)
    pairs = tree.query_pairs(R, output_type="ndarray")
    n = len(x)
    if len(pairs) == 0:
        return np.ones(n, dtype=int)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(len(rows))
    A = coo_matrix((data, (rows, cols)), shape=(n, n))
    _, labels = connected_components(A, directed=False)
    return np.bincount(labels)


def main():
    N, L, v0 = 500, 15.0, 0.05
    R_r, R_a = 0.45, 0.7
    n_warm, n_meas, n_skip = 1500, 2000, 25
    seeds = [11, 23, 41]

    alphas = np.array([1.0, 1.5, 2.0])
    etas = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.30])

    # Log bins for P(s); 30 bins from 1 to N.
    hist_bins = np.geomspace(1.0, N + 1, 31)
    n_bins = len(hist_bins) - 1
    hist = np.zeros((len(alphas), len(etas), n_bins))
    smax_frac = np.zeros((len(alphas), len(etas)))

    pbar = tqdm(total=len(alphas) * len(etas) * len(seeds),
                desc="clusters")
    for ia, alpha in enumerate(alphas):
        for ie, eta in enumerate(etas):
            sum_smax = 0.0
            n_acc = 0
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
                        sizes = cluster_sizes(sim.x, sim.y, L, R_a)
                        h, _ = np.histogram(sizes, bins=hist_bins)
                        hist[ia, ie] += h
                        sum_smax += sizes.max()
                        n_acc += 1
                pbar.update(1)
            smax_frac[ia, ie] = sum_smax / (n_acc * N)
    pbar.close()

    np.savez_compressed(
        DATA / "clusters.npz",
        alphas=alphas, etas=etas,
        hist_bins=hist_bins, hist=hist,
        smax_frac=smax_frac,
        params=np.array([N, L, v0, R_r, R_a, n_warm, n_meas, n_skip,
                         len(seeds)], dtype=float),
    )
    print("saved:", DATA / "clusters.npz")


if __name__ == "__main__":
    main()
