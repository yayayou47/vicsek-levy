"""
Topological FSS at L=64 (5 seeds, 2 alphas, 6 etas), parallelised
via ProcessPoolExecutor. Saves data/topo_L64.npz with the same
schema as topo_fss.npz but a single L slice. The merged
data/topo_fss_with_L64.npz is built by run_topo_merge.py.

Output: data/topo_L64.npz with phi/chi/binder of shape
(n_alpha, n_eta, n_seed) for L=64 only.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from topological import TopoParams, TopoVicsek


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def _measure(args):
    L, N, alpha, eta, seed, v0, R_r, k_topo, n_warm, n_meas = args
    p = TopoParams(
        N=int(N), L=float(L), v0=v0, R_r=R_r, k=k_topo,
        eta=float(eta), alpha=float(alpha), seed=int(seed),
    )
    sim = TopoVicsek(p)
    sim.theta[:] = 0.0
    for _ in range(n_warm):
        sim.step()
    phis = np.empty(n_meas)
    for j in range(n_meas):
        sim.step()
        phis[j] = sim.polarisation()
    m2 = float(np.mean(phis ** 2))
    m4 = float(np.mean(phis ** 4))
    return alpha, eta, seed, (
        float(phis.mean()),
        float(p.N * phis.var()),
        1.0 - m4 / (3.0 * m2 ** 2),
    )


def main():
    sigma = 2.22
    L = 64.0
    N = int(round(sigma * L * L))
    v0 = 0.05
    R_r = 0.45
    k_topo = 6
    n_warm, n_meas = 1200, 800
    seeds = [11, 23, 41, 67, 89]

    alphas = np.array([1.0, 2.0])
    etas = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.50])

    n_a, n_e, n_s = len(alphas), len(etas), len(seeds)
    phi = np.zeros((n_a, n_e, n_s))
    chi = np.zeros((n_a, n_e, n_s))
    binder = np.zeros((n_a, n_e, n_s))

    jobs = [
        (L, N, alpha, eta, seed, v0, R_r, k_topo, n_warm, n_meas)
        for alpha in alphas
        for eta in etas
        for seed in seeds
    ]
    n_workers = 8
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for r in tqdm(ex.map(_measure, jobs), total=len(jobs),
                      desc="topo L=64"):
            alpha, eta, seed, (ph, ch, bi) = r
            ia = int(np.where(alphas == alpha)[0][0])
            ie = int(np.where(etas == eta)[0][0])
            is_ = int(seeds.index(seed))
            phi[ia, ie, is_] = ph
            chi[ia, ie, is_] = ch
            binder[ia, ie, is_] = bi

    np.savez_compressed(
        DATA / "topo_L64.npz",
        alphas=alphas, etas=etas, L=np.float64(L), N=np.int64(N),
        seeds=np.array(seeds),
        phi=phi, chi=chi, binder=binder,
        params=np.array([sigma, v0, R_r, k_topo, n_warm, n_meas,
                         len(seeds)], dtype=float),
    )
    print("saved:", DATA / "topo_L64.npz")


if __name__ == "__main__":
    main()
