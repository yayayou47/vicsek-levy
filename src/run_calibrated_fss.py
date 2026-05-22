"""
FSS at fixed wrapped circular variance V (rather than at fixed bare eta).
This addresses the question "does the alpha-erosion of FSS persist when
the noise amplitude is calibrated?".

Method: for each alpha, calibrate eta_alpha(V) via the wrapped-Cauchy
inversion already used in run_calibrated.py, then perform the standard
metric FSS sweep at L in {15, 22, 30, 45} for V in a small grid that
brackets the FSS-located transition. Per-seed traces saved.

Output: data/fss_calibrated.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from analysis import time_series_stats
from run_calibrated import calibrate
from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def main():
    sigma = 500.0 / 15.0**2
    Ls = np.array([15.0, 22.0, 30.0, 45.0])
    Ns = (sigma * Ls**2).round().astype(int)
    R_r_abs, R_a_abs = 0.45, 0.7
    alphas = np.array([1.0, 2.0])
    # V grid: focused on the L=15 chi-peak (V ~ 0.04). The peak in V drifts
    # with L, so a refined small-V scan is essential.
    V_targets = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.12])
    n_warm, n_meas = 1500, 800
    v0 = 0.05
    seeds = np.array([0, 1, 2])

    # 1. Calibrate eta_alpha(V) on the V grid for each alpha.
    print("calibrating noise scales ...")
    rng = np.random.default_rng(0)
    eta_table = np.zeros((len(alphas), len(V_targets)))
    for ia, alpha in enumerate(alphas):
        eta_table[ia] = calibrate(float(alpha), V_targets, rng=rng)
    print("eta_table:\n", eta_table)

    # 2. FSS sweep at fixed V.
    n_L, n_a, n_v, n_s = len(Ls), len(alphas), len(V_targets), len(seeds)
    phi = np.zeros((n_L, n_a, n_v, n_s))
    chi = np.zeros_like(phi)
    U4 = np.zeros_like(phi)

    pbar = tqdm(total=n_L * n_a * n_v * n_s, desc="fss_calibrated")
    for iL, (L, N) in enumerate(zip(Ls, Ns)):
        for ia, alpha in enumerate(alphas):
            for iv, _Vt in enumerate(V_targets):
                eta_eff = float(eta_table[ia, iv])
                for isd, sd in enumerate(seeds):
                    p = VicsekParams(
                        N=int(N), L=float(L), v0=v0,
                        R_r=R_r_abs, R_a=R_a_abs,
                        eta=eta_eff, alpha=float(alpha),
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
                    phi[iL, ia, iv, isd] = s["phi_mean"]
                    chi[iL, ia, iv, isd] = s["chi"]
                    U4[iL, ia, iv, isd] = s["binder"]
                    pbar.update(1)
    pbar.close()

    np.savez_compressed(
        DATA / "fss_calibrated_v2.npz",
        Ls=Ls, Ns=Ns, alphas=alphas, V_targets=V_targets,
        eta_table=eta_table, seeds=seeds,
        phi=phi, chi=chi, U4=U4,
        sigma=np.array(sigma),
        params=np.array([v0, n_warm, n_meas], dtype=float),
    )
    print("saved:", DATA / "fss_calibrated.npz")


if __name__ == "__main__":
    main()
