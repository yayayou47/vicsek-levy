"""
Finite-size scaling: <phi>, chi, U4 vs eta for several L at fixed
number density sigma = N/L^2. Output:
   data/fss_sweep.npz  (Ls, Ns, alphas, etas, phi_mean, chi, U4, sigma)
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
    sigma = 500.0 / 15.0**2          # standard density ~ 2.222
    Ls = np.array([15.0, 22.0, 30.0, 45.0])
    Ns = (sigma * Ls**2).round().astype(int)
    # FSS requires fixed *absolute* interaction radii, otherwise the
    # number of neighbours within R_r grows like (rL)^2 sigma and the
    # dynamics is no longer scale-invariant. We hold R_r = 0.5 and
    # R_a = 0.7 (the standard L=15 values) and adjust r, a per L.
    R_r_abs, R_a_abs = 0.5, 0.7
    alphas = np.array([1.0, 2.0])
    etas = np.linspace(0.0, 0.4, 17)
    # Relaxation must outlast one box crossing (L/v0 steps); aligned
    # initial condition lets the warmup only damp fluctuations.
    n_warm, n_meas = 1500, 800
    v0, seed = 0.05, 12345

    n_L, n_a, n_e = len(Ls), len(alphas), len(etas)
    phi_mean = np.zeros((n_L, n_a, n_e))
    chi = np.zeros_like(phi_mean)
    U4 = np.zeros_like(phi_mean)

    pbar = tqdm(total=n_L * n_a * n_e, desc="fss")
    for iL, (L, N) in enumerate(zip(Ls, Ns)):
        for ia, alpha in enumerate(alphas):
            for ie, eta in enumerate(etas):
                p = VicsekParams(
                    N=int(N), L=float(L), v0=v0,
                    R_r=R_r_abs, R_a=R_a_abs,
                    eta=float(eta), alpha=float(alpha), seed=seed,
                )
                sim = Vicsek(p)
                sim.theta[:] = 0.0  # aligned initial condition
                for _ in range(n_warm):
                    sim.step()
                phi = np.empty(n_meas)
                for k in range(n_meas):
                    sim.step()
                    phi[k] = sim.polarisation()
                stats = time_series_stats(phi, p.N)
                phi_mean[iL, ia, ie] = stats["phi_mean"]
                chi[iL, ia, ie] = stats["chi"]
                U4[iL, ia, ie] = stats["binder"]
                pbar.update(1)
    pbar.close()

    np.savez_compressed(
        DATA / "fss_sweep.npz",
        Ls=Ls, Ns=Ns, alphas=alphas, etas=etas,
        phi_mean=phi_mean, chi=chi, U4=U4,
        sigma=np.array(sigma),
        params=np.array([v0, n_warm, n_meas], dtype=float),
    )
    print("saved:", DATA / "fss_sweep.npz")


if __name__ == "__main__":
    main()
