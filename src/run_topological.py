"""
Mini-FSS for the topological Vicsek-Couzin variant. Sizes
L in {15, 22, 30, 45} at fixed density sigma = 2.22, two stability
indices alpha = 1, 2, k = 6 topological alignment neighbours. We
sweep eta across the transition and record polar order,
susceptibility, and the Binder cumulant per seed (5 seeds), so
downstream code can draw SEM bands and bootstrap slope CIs.

Output: data/topo_fss.npz
   alphas, etas, Ls, Ns, seeds
   phi[a, l, e, s], chi[a, l, e, s], binder[a, l, e, s]
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
    return L, alpha, eta, seed, (
        float(phis.mean()),
        float(p.N * phis.var()),
        1.0 - m4 / (3.0 * m2 ** 2),
    )


def main():
    sigma = 2.22
    Ls = np.array([15.0, 22.0, 30.0, 45.0])
    v0 = 0.05
    R_r = 0.45
    k_topo = 6
    n_warm, n_meas = 1200, 800
    seeds = [11, 23, 41, 67, 89]

    alphas = np.array([1.0, 2.0])
    etas = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.50])

    n_a, n_l, n_e, n_s = len(alphas), len(Ls), len(etas), len(seeds)
    phi = np.zeros((n_a, n_l, n_e, n_s))
    chi = np.zeros((n_a, n_l, n_e, n_s))
    binder = np.zeros((n_a, n_l, n_e, n_s))

    Ns = np.array([int(round(sigma * L * L)) for L in Ls])

    jobs = [
        (Ls[il], Ns[il], alpha, eta, seed,
         v0, R_r, k_topo, n_warm, n_meas)
        for ia, alpha in enumerate(alphas)
        for il in range(n_l)
        for eta in etas
        for seed in seeds
    ]
    n_workers = 8
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for r in tqdm(ex.map(_measure, jobs), total=len(jobs),
                      desc="topo"):
            L, alpha, eta, seed, (ph, ch, bi) = r
            ia = int(np.where(alphas == alpha)[0][0])
            il = int(np.where(Ls == L)[0][0])
            ie = int(np.where(etas == eta)[0][0])
            is_ = int(seeds.index(seed))
            phi[ia, il, ie, is_] = ph
            chi[ia, il, ie, is_] = ch
            binder[ia, il, ie, is_] = bi

    np.savez_compressed(
        DATA / "topo_fss.npz",
        alphas=alphas, etas=etas, Ls=Ls, Ns=Ns,
        seeds=np.array(seeds),
        phi=phi, chi=chi, binder=binder,
        params=np.array([sigma, v0, R_r, k_topo, n_warm, n_meas,
                         len(seeds)], dtype=float),
    )
    print("saved:", DATA / "topo_fss.npz")


if __name__ == "__main__":
    main()
