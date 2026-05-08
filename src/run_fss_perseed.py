"""
Per-seed FSS sweep -- same L grid as the published fss_sweep.npz but
saving the polarisation traces seed-by-seed so that bootstrap error
bars can be computed on chi_max(L) and on the FSS slopes. Output:
   data/fss_perseed.npz
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
    Ls = np.array([15.0, 22.0, 30.0, 45.0])
    Ns = (sigma * Ls**2).round().astype(int)
    R_r_abs, R_a_abs = 0.5, 0.7
    alphas = np.array([1.0, 2.0])
    etas = np.linspace(0.0, 0.4, 17)
    n_warm, n_meas = 1500, 800
    v0 = 0.05
    seeds = np.array([0, 1, 2, 3, 4])

    n_L, n_a, n_e, n_s = len(Ls), len(alphas), len(etas), len(seeds)
    phi = np.zeros((n_L, n_a, n_e, n_s))
    chi = np.zeros_like(phi)
    U4 = np.zeros_like(phi)

    pbar = tqdm(total=n_L * n_a * n_e * n_s, desc="fss_perseed")
    for iL, (L, N) in enumerate(zip(Ls, Ns)):
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
                    phi[iL, ia, ie, isd] = s["phi_mean"]
                    chi[iL, ia, ie, isd] = s["chi"]
                    U4[iL, ia, ie, isd] = s["binder"]
                    pbar.update(1)
    pbar.close()

    np.savez_compressed(
        DATA / "fss_perseed.npz",
        Ls=Ls, Ns=Ns, alphas=alphas, etas=etas, seeds=seeds,
        phi=phi, chi=chi, U4=U4,
        sigma=np.array(sigma),
        params=np.array([v0, n_warm, n_meas], dtype=float),
    )
    print("saved:", DATA / "fss_perseed.npz")


if __name__ == "__main__":
    main()
