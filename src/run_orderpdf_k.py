"""
Order-parameter pdf at the k-dependent topological-Cauchy transition.
The k-scan of run_topo_k.py shows that the susceptibility slope
ranges from 0.85 (k=4) to 2.15 (k=10) at alpha = 1. Slope ~ 2 is
the clean first-order value. We test whether the k = 10 endpoint is
genuinely first-order (bimodal P(phi)) or merely a sharper
continuous transition (unimodal P(phi)). The k = 4 case is
included for completeness.

Output: data/orderpdf_k.npz
   labels[k], phi_traj[k, t]  (concatenated over seeds)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from topological import TopoParams, TopoVicsek


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def trace_phi(p: TopoParams, n_warm: int, n_meas: int):
    sim = TopoVicsek(p)
    sim.theta[:] = 0.0
    for _ in range(n_warm):
        sim.step()
    out = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        out[k] = sim.polarisation()
    return out


def main():
    L = 30.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    v0 = 0.05
    R_r = 0.45
    n_warm = 2000
    n_meas = 8000
    seeds = [11, 23, 41]

    # k -> near-transition eta (from run_topo_k.py L=30 chi peaks).
    cases = [
        ("k4",  4,  0.15),
        ("k10", 10, 0.30),
    ]

    n_cases = len(cases)
    phi_traj = np.zeros((n_cases, len(seeds) * n_meas))
    labels = np.array([c[0] for c in cases])

    pbar = tqdm(total=n_cases * len(seeds), desc="orderpdf_k")
    for ic, (lbl, k_topo, eta) in enumerate(cases):
        traj_list = []
        for seed in seeds:
            p = TopoParams(
                N=N, L=L, v0=v0, R_r=R_r, k=int(k_topo),
                eta=float(eta),
                alpha=1.0, seed=seed,
            )
            traj_list.append(trace_phi(p, n_warm, n_meas))
            pbar.update(1)
        phi_traj[ic] = np.concatenate(traj_list)
    pbar.close()

    np.savez_compressed(
        DATA / "orderpdf_k.npz",
        labels=labels, phi_traj=phi_traj,
        meta=np.array([[c[1], c[2]] for c in cases], dtype=float),
        params=np.array([N, L, v0, R_r, n_warm, n_meas,
                         len(seeds)], dtype=float),
    )
    print("saved:", DATA / "orderpdf_k.npz")


if __name__ == "__main__":
    main()
