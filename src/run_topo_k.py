"""
Robustness of the topological-Cauchy susceptibility scaling under
variation of the topological neighbour count k. We measure
chi_max(L=15) vs chi_max(L=30) at alpha = 1 for k in {4, 6, 10}; the
ratio probes whether the slope chi_max ~ L^{1.58} is k-universal.

Output: data/topo_k_scan.npz
   ks, Ls, etas, chi[k, L, e], phi[k, L, e]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from topological import TopoParams, TopoVicsek


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def measure(p: TopoParams, n_warm: int, n_meas: int):
    sim = TopoVicsek(p)
    sim.theta[:] = 0.0
    for _ in range(n_warm):
        sim.step()
    phis = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        phis[k] = sim.polarisation()
    return phis.mean(), p.N * phis.var()


def main():
    sigma = 2.22
    Ls = np.array([15.0, 30.0])
    v0 = 0.05
    R_r = 0.45
    n_warm, n_meas = 1200, 800
    seeds = [11, 23]

    ks = np.array([4, 6, 10])
    etas = np.array([0.10, 0.15, 0.20, 0.25, 0.30])

    n_k, n_l, n_e = len(ks), len(Ls), len(etas)
    phi = np.zeros((n_k, n_l, n_e))
    chi = np.zeros((n_k, n_l, n_e))

    Ns = np.array([int(round(sigma * L * L)) for L in Ls])

    pbar = tqdm(total=n_k * n_l * n_e * len(seeds), desc="topo_k")
    for ik, k_topo in enumerate(ks):
        for il, L in enumerate(Ls):
            for ie, eta in enumerate(etas):
                p_acc = 0.0
                c_acc = 0.0
                for seed in seeds:
                    p = TopoParams(
                        N=int(Ns[il]), L=float(L), v0=v0,
                        R_r=R_r, k=int(k_topo),
                        eta=float(eta),
                        alpha=1.0, seed=seed,
                    )
                    ph, ch = measure(p, n_warm, n_meas)
                    p_acc += ph
                    c_acc += ch
                    pbar.update(1)
                phi[ik, il, ie] = p_acc / len(seeds)
                chi[ik, il, ie] = c_acc / len(seeds)
    pbar.close()

    np.savez_compressed(
        DATA / "topo_k_scan.npz",
        ks=ks, Ls=Ls, Ns=Ns, etas=etas,
        phi=phi, chi=chi,
        params=np.array([sigma, v0, R_r, n_warm, n_meas,
                         len(seeds)], dtype=float),
    )

    # Quick report.
    print()
    print("chi_max scaling by k (alpha = 1):")
    print(f"{'k':>3} {'chi(15)':>9} {'chi(30)':>9} {'slope':>8}")
    for ik, k_t in enumerate(ks):
        c15 = chi[ik, 0].max()
        c30 = chi[ik, 1].max()
        s = np.log(c30 / c15) / np.log(30.0 / 15.0)
        print(f"{int(k_t):>3} {c15:>9.2f} {c30:>9.2f} {s:>8.3f}")
    print("saved:", DATA / "topo_k_scan.npz")


if __name__ == "__main__":
    main()
