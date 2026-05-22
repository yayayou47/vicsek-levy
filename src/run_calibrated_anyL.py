"""
V-calibrated FSS at a single user-specified L (5 seeds, 2 alphas,
8 V targets). Same schema as run_calibrated_L64.py but L is a CLI
argument so the same driver covers L = 91 and L = 128.

Output: data/fss_calibrated_L{L}.npz with phi/chi/U4 of shape
(n_alpha, n_v, n_seed) for the chosen L only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from analysis import time_series_stats
from run_calibrated import calibrate
from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" if HERE.parent.name != "notes" else \
       HERE.parent.parent / "data"
DATA.mkdir(exist_ok=True)


def _run_one(args):
    N, L, v0, R_r, R_a, eta, alpha, seed, n_warm, n_meas = args
    p = VicsekParams(
        N=int(N), L=float(L), v0=v0,
        R_r=R_r, R_a=R_a,
        eta=float(eta), alpha=float(alpha), seed=int(seed),
    )
    sim = Vicsek(p)
    sim.theta[:] = 0.0
    for _ in range(n_warm):
        sim.step()
    trace = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        trace[k] = sim.polarisation()
    s = time_series_stats(trace, p.N)
    return alpha, eta, seed, (s["phi_mean"], s["chi"], s["binder"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=float, required=True,
                        help="box side L (must match sigma=500/15^2 protocol)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    L = float(args.L)
    sigma = 500.0 / 15.0**2
    N = int(round(sigma * L * L))
    R_r_abs, R_a_abs = 0.45, 0.7
    v0 = 0.05
    n_warm, n_meas = 1500, 800
    seeds = [0, 1, 2, 3, 4]

    alphas = np.array([1.0, 2.0])
    V_targets = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.12])

    print(f"L = {L}, N = {N}, sigma = {sigma:.4f}")
    print("calibrating noise scales ...")
    rng = np.random.default_rng(0)
    eta_table = np.zeros((len(alphas), len(V_targets)))
    for ia, alpha in enumerate(alphas):
        eta_table[ia] = calibrate(float(alpha), V_targets, rng=rng)
    print("eta_table:\n", eta_table)

    n_a, n_v, n_s = len(alphas), len(V_targets), len(seeds)
    phi = np.zeros((n_a, n_v, n_s))
    chi = np.zeros_like(phi)
    U4 = np.zeros_like(phi)

    jobs = []
    for ia, alpha in enumerate(alphas):
        for iv, V_t in enumerate(V_targets):
            eta_eff = float(eta_table[ia, iv])
            for isd, sd in enumerate(seeds):
                jobs.append(
                    (N, L, v0, R_r_abs, R_a_abs,
                     eta_eff, float(alpha), int(sd), n_warm, n_meas)
                )

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in tqdm(ex.map(_run_one, jobs), total=len(jobs),
                      desc=f"V-calibrated L={int(L)}"):
            alpha, eta, seed, (ph, ch, bi) = r
            ia = int(np.where(alphas == alpha)[0][0])
            iv = int(np.argmin(np.abs(eta_table[ia] - eta)))
            isd = seeds.index(seed)
            phi[ia, iv, isd] = ph
            chi[ia, iv, isd] = ch
            U4[ia, iv, isd] = bi

    out = DATA / f"fss_calibrated_L{int(L)}.npz"
    np.savez_compressed(
        out,
        L=np.float64(L), N=np.int64(N), alphas=alphas,
        V_targets=V_targets, eta_table=eta_table,
        seeds=np.array(seeds),
        phi=phi, chi=chi, U4=U4,
        sigma=np.array(sigma),
        params=np.array([v0, n_warm, n_meas], dtype=float),
    )
    print("saved:", out)


if __name__ == "__main__":
    main()
