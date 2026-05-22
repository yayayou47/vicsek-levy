"""
Topological FSS at a single user-specified L, parametric over the
alpha grid so it can produce both the legacy 2-alpha output and the
harmonised 3-alpha {1, 1.5, 2} output without code duplication.

Output: data/topo_L{L}.npz (default) or data/topo_L{L}_a3.npz when
--alphas covers three values.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from topological import TopoParams, TopoVicsek


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" if HERE.parent.name != "notes" else \
       HERE.parent.parent / "data"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=float, required=True)
    parser.add_argument("--alphas", type=str, default="1.0,2.0",
                        help="comma-separated alpha grid")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    L = float(args.L)
    alphas = np.array([float(a) for a in args.alphas.split(",")])
    sigma = 2.22
    N = int(round(sigma * L * L))
    v0 = 0.05
    R_r = 0.45
    k_topo = 6
    n_warm, n_meas = 1200, 800
    seeds = [11, 23, 41, 67, 89]

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
    print(f"L = {L}, N = {N}, alphas = {alphas.tolist()}, "
          f"|jobs| = {len(jobs)}")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in tqdm(ex.map(_measure, jobs), total=len(jobs),
                      desc=f"topo L={int(L)}"):
            alpha, eta, seed, (ph, ch, bi) = r
            ia = int(np.where(alphas == alpha)[0][0])
            ie = int(np.where(etas == eta)[0][0])
            is_ = int(seeds.index(seed))
            phi[ia, ie, is_] = ph
            chi[ia, ie, is_] = ch
            binder[ia, ie, is_] = bi

    # Suffix the output filename so a partial-alpha run never
    # overwrites a denser sibling. _a3 means the full {1, 1.5, 2} grid;
    # otherwise the alpha list itself is appended (e.g. _a1.5).
    if len(alphas) >= 3:
        suffix = "_a3"
    else:
        suffix = "_a" + "_".join(f"{float(a):g}" for a in alphas)
    out = DATA / f"topo_L{int(L)}{suffix}.npz"
    np.savez_compressed(
        out,
        L=np.float64(L), N=np.int64(N), alphas=alphas,
        etas=etas, seeds=np.array(seeds),
        phi=phi, chi=chi, U4=binder,
        params=np.array([v0, R_r, k_topo, n_warm, n_meas],
                        dtype=float),
    )
    print("saved:", out)


if __name__ == "__main__":
    main()
