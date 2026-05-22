"""
Extend the existing topo_fss.npz (alphas in {1, 2}, L in {15..45})
to a third alpha slice alpha = 1.5 so the topological FSS lever
matches the metric / GNF / bands / MSD analyses. Writes
data/topo_fss_a15.npz with only the alpha = 1.5 slice; the
make_figures topological builder concatenates the two files.

Output: data/topo_fss_a15.npz with phi/chi/U4 of shape
(n_L, n_eta, n_seed) at alpha = 1.5 only.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from topological import TopoParams, TopoVicsek


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" if HERE.parent.name != "notes" else \
       HERE.parent.parent / "data"


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
    return L, eta, seed, (
        float(phis.mean()),
        float(p.N * phis.var()),
        1.0 - m4 / (3.0 * m2 ** 2),
    )


def main():
    sigma = 2.22
    Ls = np.array([15.0, 22.0, 30.0, 45.0])
    Ns = (sigma * Ls ** 2).round().astype(int)
    v0 = 0.05
    R_r = 0.45
    k_topo = 6
    n_warm, n_meas = 1200, 800
    seeds = [11, 23, 41, 67, 89]
    alpha = 1.5
    etas = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.50])

    n_L, n_e, n_s = len(Ls), len(etas), len(seeds)
    phi = np.zeros((n_L, n_e, n_s))
    chi = np.zeros((n_L, n_e, n_s))
    binder = np.zeros((n_L, n_e, n_s))

    jobs = [
        (float(L), int(N), float(alpha), float(eta), int(seed),
         v0, R_r, k_topo, n_warm, n_meas)
        for L, N in zip(Ls, Ns)
        for eta in etas
        for seed in seeds
    ]
    print(f"alpha = {alpha}, Ls = {Ls.tolist()}, |jobs| = {len(jobs)}")

    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in tqdm(ex.map(_measure, jobs), total=len(jobs),
                      desc="topo a=1.5 smallL"):
            L_done, eta, seed, (ph, ch, bi) = r
            iL = int(np.where(Ls == L_done)[0][0])
            ie = int(np.where(etas == eta)[0][0])
            is_ = int(seeds.index(seed))
            phi[iL, ie, is_] = ph
            chi[iL, ie, is_] = ch
            binder[iL, ie, is_] = bi

    out = DATA / "topo_fss_a15.npz"
    np.savez_compressed(
        out,
        Ls=Ls, Ns=Ns,
        alpha=np.float64(alpha),
        etas=etas, seeds=np.array(seeds),
        phi=phi, chi=chi, U4=binder,
        params=np.array([v0, R_r, k_topo, n_warm, n_meas],
                        dtype=float),
    )
    print("saved:", out)


if __name__ == "__main__":
    main()
