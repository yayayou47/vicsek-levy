"""
Phase-diagram robustness across the other "fixed" parameters of the
two-zone Vicsek model: speed v_0 and density sigma. (The blind-angle
sweep has been dropped with the blind sector itself; the R_r scan
that fixed R_r = 0.45 lives in run_noblind_scan.py.)
Per-seed measurements are kept so SEM bands can be drawn downstream.
5 seeds per (alpha, axis-value, eta) point, parallelised via
ProcessPoolExecutor.

Output: data/robustness.npz
   v0_grid, sigma_grid
   phi_v0[a, v, e, s], phi_sigma[a, sg, e, s]
   etas, alphas, seeds, base_params
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def _measure_phi(args):
    """Worker for a single (axis_name, val, alpha, eta, seed) point."""
    axis_name, val, alpha, eta, seed, base, n_warm, n_meas, L = args
    kwargs = dict(base)
    if axis_name == "v0":
        kwargs["v0"] = float(val)
    elif axis_name == "sigma":
        sigma = float(val)
        kwargs["N"] = int(round(sigma * L * L))
    kwargs["eta"] = float(eta)
    kwargs["alpha"] = float(alpha)
    kwargs["seed"] = int(seed)
    p = VicsekParams(**kwargs)
    sim = Vicsek(p)
    sim.theta[:] = 0.0
    for _ in range(n_warm):
        sim.step()
    s = 0.0
    for _ in range(n_meas):
        sim.step()
        s += sim.polarisation()
    return axis_name, val, alpha, eta, seed, s / n_meas


def _sweep(axis_name, grid, alphas, etas, seeds, base, n_warm, n_meas, L,
           n_workers):
    n_a, n_g, n_e, n_s = len(alphas), len(grid), len(etas), len(seeds)
    out = np.zeros((n_a, n_g, n_e, n_s))
    jobs = [
        (axis_name, val, alpha, eta, seed, base, n_warm, n_meas, L)
        for alpha in alphas
        for val in grid
        for eta in etas
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for r in tqdm(ex.map(_measure_phi, jobs), total=len(jobs),
                      desc=axis_name):
            _, val, alpha, eta, seed, phi = r
            ia = int(np.where(alphas == alpha)[0][0])
            ig = int(np.where(grid == val)[0][0])
            ie = int(np.where(etas == eta)[0][0])
            is_ = int(seeds.index(seed))
            out[ia, ig, ie, is_] = phi
    return out


def main():
    L = 15.0
    R_r, R_a = 0.45, 0.7
    n_warm, n_meas = 1200, 800
    seeds = [11, 23, 41, 67, 89]

    alphas = np.array([1.0, 2.0])
    etas = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.30])

    v0_grid = np.array([0.02, 0.05, 0.10, 0.20])
    sigma_grid = np.array([1.5, 2.22, 3.0, 4.5])

    sigma0 = 2.22
    N0 = int(round(sigma0 * L * L))
    base = dict(N=N0, L=L, v0=0.05, R_r=R_r, R_a=R_a)

    n_workers = 8

    phi_v0 = _sweep("v0", v0_grid, alphas, etas, seeds, base,
                    n_warm, n_meas, L, n_workers)
    phi_sigma = _sweep("sigma", sigma_grid, alphas, etas, seeds, base,
                       n_warm, n_meas, L, n_workers)

    np.savez_compressed(
        DATA / "robustness.npz",
        v0_grid=v0_grid, sigma_grid=sigma_grid,
        phi_v0=phi_v0, phi_sigma=phi_sigma,
        etas=etas, alphas=alphas, seeds=np.array(seeds),
        base_params=np.array(
            [N0, L, 0.05, R_r, R_a, n_warm, n_meas,
             len(seeds)], dtype=float),
    )
    print("saved:", DATA / "robustness.npz")


if __name__ == "__main__":
    main()
