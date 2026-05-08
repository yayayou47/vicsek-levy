"""
FSS extension to L=64 for alpha in {1, 2} on a refined eta grid bracketing
the FSS-located eta_c. This addresses the "is alpha=2 actually first-order?"
question by adding a fifth size to the FSS slope and letting the band-channel
mature. Output:
   data/fss_L64.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from analysis import time_series_stats
from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def main():
    sigma = 500.0 / 15.0**2
    L = 64.0
    N = int(round(sigma * L * L))
    R_r_abs, R_a_abs = 0.5, 0.7
    alphas = np.array([1.0, 2.0])
    # Refined grid around the FSS eta_c per alpha
    etas = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.24, 0.30])
    n_warm, n_meas = 2000, 1200
    v0 = 0.05
    seeds = np.array([0, 1, 2, 3, 4])

    n_a, n_e, n_s = len(alphas), len(etas), len(seeds)
    phi = np.zeros((n_a, n_e, n_s))
    chi = np.zeros_like(phi)
    U4 = np.zeros_like(phi)

    pbar = tqdm(total=n_a * n_e * n_s, desc=f"fss_L{int(L)}")
    for ia, alpha in enumerate(alphas):
        for ie, eta in enumerate(etas):
            for isd, sd in enumerate(seeds):
                p = VicsekParams(
                    N=int(N), L=float(L), v0=v0,
                    R_r=R_r_abs, R_a=R_a_abs,
                    eta=float(eta), alpha=float(alpha),
                    seed=int(sd),
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
                phi[ia, ie, isd] = s["phi_mean"]
                chi[ia, ie, isd] = s["chi"]
                U4[ia, ie, isd] = s["binder"]
                pbar.update(1)
    pbar.close()

    np.savez_compressed(
        DATA / "fss_L64.npz",
        L=np.array(L), N=np.array(N),
        alphas=alphas, etas=etas, seeds=seeds,
        phi=phi, chi=chi, U4=U4,
        sigma=np.array(sigma),
        params=np.array([v0, n_warm, n_meas], dtype=float),
    )
    print("saved:", DATA / "fss_L64.npz")


if __name__ == "__main__":
    main()
