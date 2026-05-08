"""
3D phase sweep: polar order phi(eta, R_r, R_a) at three values of the
Levy stability index alpha. Both interaction radii are absolute.
Output:
   data/sweep_3d.npz  with axes (alphas, etas, R_rs, R_as) and arrays
   phi[ia, ie, ir, iA] and chi[ia, ie, ir, iA].
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
    N, L, v0, seed = 400, 15.0, 0.05, 42
    n_warm, n_meas = 400, 400

    alphas = np.array([1.0, 1.5, 2.0])
    etas = np.linspace(0.0, 0.4, 9)
    R_rs = np.linspace(0.20, 0.80, 7)
    R_as = np.linspace(0.30, 1.20, 7)

    n_a, n_e, n_r, n_A = len(alphas), len(etas), len(R_rs), len(R_as)
    phi = np.full((n_a, n_e, n_r, n_A), np.nan)
    chi = np.full_like(phi, np.nan)

    pbar = tqdm(total=n_a * n_e * n_r * n_A, desc="3d")
    for ia, alpha in enumerate(alphas):
        for ie, eta in enumerate(etas):
            for ir, R_r in enumerate(R_rs):
                for iA, R_a in enumerate(R_as):
                    if R_a <= R_r:
                        # Alignment annulus empty -> simulation undefined
                        # in the zonal sense. Skip; leave NaN.
                        pbar.update(1)
                        continue
                    p = VicsekParams(
                        N=N, L=L, v0=v0,
                        R_r=float(R_r), R_a=float(R_a),
                        eta=float(eta), alpha=float(alpha), seed=seed,
                    )
                    sim = Vicsek(p)
                    sim.theta[:] = 0.0  # aligned init for fast warmup
                    for _ in range(n_warm):
                        sim.step()
                    samples = np.empty(n_meas)
                    for k in range(n_meas):
                        sim.step()
                        samples[k] = sim.polarisation()
                    stats = time_series_stats(samples, p.N)
                    phi[ia, ie, ir, iA] = stats["phi_mean"]
                    chi[ia, ie, ir, iA] = stats["chi"]
                    pbar.update(1)
    pbar.close()

    np.savez_compressed(
        DATA / "sweep_3d.npz",
        alphas=alphas, etas=etas, R_rs=R_rs, R_as=R_as,
        phi=phi, chi=chi,
        params=np.array([N, L, v0, n_warm, n_meas], dtype=float),
    )
    print("saved:", DATA / "sweep_3d.npz")


if __name__ == "__main__":
    main()
