"""
Probability density of the polar order parameter at the FSS-located
near-transition point, in four corners of model x noise:

  metric, alpha = 2  : known first-order; pdf should be bimodal
  metric, alpha = 1  : we found L-trivial signatures; pdf should be unimodal
  topological, alpha = 2 : Ginelli-Chate continuous; unimodal expected
  topological, alpha = 1 : new chi-divergent transition; pdf shape is the
                           direct test of its order

Output: data/orderpdf.npz
   labels[k], phi_traj[k, t]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from vicsek import Vicsek, VicsekParams
from topological import TopoVicsek, TopoParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def trace_phi(sim_factory, n_warm, n_meas):
    sim = sim_factory()
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
    R_r, R_a = 0.5, 0.7
    k_topo = 6
    n_warm = 2000
    n_meas = 8000
    seeds = [11, 23, 41]

    # (label, factory)
    cases = [
        ("metric_a2",
            lambda s: Vicsek(VicsekParams(
                N=N, L=L, v0=v0, R_r=R_r, R_a=R_a,
                eta=0.15, alpha=2.0, seed=s))),
        ("metric_a1",
            lambda s: Vicsek(VicsekParams(
                N=N, L=L, v0=v0, R_r=R_r, R_a=R_a,
                eta=0.05, alpha=1.0, seed=s))),
        ("topo_a2",
            lambda s: TopoVicsek(TopoParams(
                N=N, L=L, v0=v0, R_r=R_r, k=k_topo,
                eta=0.50, alpha=2.0, seed=s))),
        ("topo_a1",
            lambda s: TopoVicsek(TopoParams(
                N=N, L=L, v0=v0, R_r=R_r, k=k_topo,
                eta=0.20, alpha=1.0, seed=s))),
    ]

    n_cases = len(cases)
    phi_traj = np.zeros((n_cases, len(seeds) * n_meas))
    labels = np.array([c[0] for c in cases])

    pbar = tqdm(total=n_cases * len(seeds), desc="orderpdf")
    for ic, (lbl, factory) in enumerate(cases):
        all_traj = []
        for seed in seeds:
            t = trace_phi(lambda s=seed: factory(s), n_warm, n_meas)
            all_traj.append(t)
            pbar.update(1)
        phi_traj[ic] = np.concatenate(all_traj)
    pbar.close()

    np.savez_compressed(
        DATA / "orderpdf.npz",
        labels=labels, phi_traj=phi_traj,
        params=np.array([N, L, v0, R_r, R_a, k_topo,
                         n_warm, n_meas, len(seeds)], dtype=float),
    )
    print("saved:", DATA / "orderpdf.npz")


if __name__ == "__main__":
    main()
