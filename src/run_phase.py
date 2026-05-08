"""
Sweep the (alpha, eta) plane and store time series of polarisation.

Output: data/phase_sweep.npz containing
   alphas, etas, phi_mean[a, e], chi[a, e], binder[a, e], phi_traj[a, e, t]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from analysis import time_series_stats
from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def run_one(alpha: float, eta: float, p: VicsekParams, n_warm: int, n_meas: int):
    p = VicsekParams(**{**p.__dict__, "alpha": alpha, "eta": eta})
    sim = Vicsek(p)
    for _ in range(n_warm):
        sim.step()
    phi = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        phi[k] = sim.polarisation()
    stats = time_series_stats(phi, p.N)
    return phi, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", nargs="+", type=float, default=[2.0, 1.5, 1.0])
    ap.add_argument("--eta-min", type=float, default=0.0)
    ap.add_argument("--eta-max", type=float, default=1.0)
    ap.add_argument("--n-eta", type=int, default=21)
    ap.add_argument("--N", type=int, default=1500)
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--v0", type=float, default=0.05)
    ap.add_argument("--n-warm", type=int, default=600)
    ap.add_argument("--n-meas", type=int, default=600)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=str, default=str(DATA / "phase_sweep.npz"))
    args = ap.parse_args()

    alphas = np.array(args.alpha, dtype=float)
    etas = np.linspace(args.eta_min, args.eta_max, args.n_eta)

    phi_mean = np.zeros((len(alphas), len(etas)))
    chi = np.zeros_like(phi_mean)
    binder = np.zeros_like(phi_mean)
    phi_traj = np.zeros((len(alphas), len(etas), args.n_meas))

    base = VicsekParams(
        N=args.N, L=args.L, v0=args.v0, seed=args.seed
    )

    pbar = tqdm(total=len(alphas) * len(etas), desc="sweep")
    for ia, a in enumerate(alphas):
        for ie, e in enumerate(etas):
            phi, stats = run_one(a, e, base, args.n_warm, args.n_meas)
            phi_mean[ia, ie] = stats["phi_mean"]
            chi[ia, ie] = stats["chi"]
            binder[ia, ie] = stats["binder"]
            phi_traj[ia, ie] = phi
            pbar.update(1)
    pbar.close()

    np.savez_compressed(
        args.out,
        alphas=alphas,
        etas=etas,
        phi_mean=phi_mean,
        chi=chi,
        binder=binder,
        phi_traj=phi_traj,
        params=np.array(
            [args.N, args.L, args.v0, args.n_warm, args.n_meas], dtype=float
        ),
    )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
