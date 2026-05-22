"""Snapshots illustrating the §order section of the manuscript.

Runs 6 configurations (2 alphas x 3 noise regimes: ordered, near-critical,
disordered) on the canonical two-zone Vicsek-Couzin model and saves the
particle positions and headings to data/order_snapshots.npz so the figure
script can render them as arrow plots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
V1_ROOT = HERE.parent.parent if HERE.parent.name == "notes" else HERE.parent
DATA = V1_ROOT / "data"


# (alpha, eta, label) per case. eta values picked relative to the
# bare-eta phase boundary measured in fig_phase_curve (eta_c ~ 0.04 at
# alpha=1, eta_c ~ 0.15 at alpha=2).
CASES = [
    (1.0, 0.01, "alpha=1 ordered"),
    (1.0, 0.04, "alpha=1 near critical"),
    (1.0, 0.15, "alpha=1 disordered"),
    (2.0, 0.05, "alpha=2 ordered"),
    (2.0, 0.15, "alpha=2 near critical"),
    (2.0, 0.30, "alpha=2 disordered"),
]


def main(L: float = 91.0, sigma: float = 2.22,
         n_warm: int = 4000, seed: int = 0) -> None:
    N = int(round(sigma * L * L))
    alphas = np.array([c[0] for c in CASES])
    etas = np.array([c[1] for c in CASES])
    snap_x = np.zeros((len(CASES), N))
    snap_y = np.zeros((len(CASES), N))
    snap_theta = np.zeros((len(CASES), N))
    phi_final = np.zeros(len(CASES))

    for ic, (alpha, eta, label) in enumerate(CASES):
        p = VicsekParams(N=N, L=L, v0=0.05, eta=eta, alpha=alpha,
                         seed=seed)
        sim = Vicsek(p)
        # Aligned initial condition (theta_i = 0), as in the FSS
        # protocol of §sec:fss: at large L the random IC does not
        # nucleate global order within a few thousand steps, so
        # ordered runs would otherwise look disordered.
        sim.theta[:] = 0.0
        for _ in range(n_warm):
            sim.step()
        snap_x[ic] = sim.x.copy()
        snap_y[ic] = sim.y.copy()
        snap_theta[ic] = sim.theta.copy()
        phi_final[ic] = sim.polarisation()
        print(f"[{ic+1}/{len(CASES)}] {label:30s} "
              f"phi={phi_final[ic]:.3f}")

    out = DATA / "order_snapshots.npz"
    np.savez(out,
             alphas=alphas, etas=etas,
             snap_x=snap_x, snap_y=snap_y, snap_theta=snap_theta,
             phi_final=phi_final,
             params=np.array([N, L, 0.05, 0.45, 0.7,
                              n_warm, seed], dtype=float))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
