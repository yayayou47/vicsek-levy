"""
Smooth phase boundary eta_c(alpha) for the metric Vicsek-Couzin model.
Sweep alpha in 6 points x eta in 18 points at L = 15, with 3 seeds.
For each alpha, locate eta_c by the chi peak (interpolated) and the
midpoint of phi(eta) (where phi crosses 1/2).

Output: data/phase_curve.npz
   alphas, etas, phi_mean[a, e], chi[a, e],
   eta_c_chi[a], eta_c_phi[a]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def measure(p: VicsekParams, n_warm: int, n_meas: int):
    sim = Vicsek(p)
    sim.theta[:] = 0.0
    for _ in range(n_warm):
        sim.step()
    phis = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        phis[k] = sim.polarisation()
    return phis.mean(), p.N * phis.var()


def main():
    L = 15.0
    sigma = 2.22
    N = int(round(sigma * L * L))
    v0 = 0.05
    R_r, R_a = 0.45, 0.7
    n_warm, n_meas = 1500, 1200
    seeds = [11, 23, 41]

    alphas = np.array([1.0, 1.25, 1.5, 1.75, 1.9, 2.0])
    etas = np.linspace(0.02, 0.40, 18)

    n_a, n_e = len(alphas), len(etas)
    phi = np.zeros((n_a, n_e))
    chi = np.zeros((n_a, n_e))

    pbar = tqdm(total=n_a * n_e * len(seeds), desc="phase_curve")
    for ia, alpha in enumerate(alphas):
        for ie, eta in enumerate(etas):
            p_acc = 0.0
            c_acc = 0.0
            for seed in seeds:
                p = VicsekParams(
                    N=N, L=L, v0=v0, R_r=R_r, R_a=R_a,
                    eta=float(eta), alpha=float(alpha), seed=seed,
                )
                ph, ch = measure(p, n_warm, n_meas)
                p_acc += ph
                c_acc += ch
                pbar.update(1)
            phi[ia, ie] = p_acc / len(seeds)
            chi[ia, ie] = c_acc / len(seeds)
    pbar.close()

    # Phase boundary: chi-peak interpolation + phi-half crossing.
    eta_c_chi = np.zeros(n_a)
    eta_c_phi = np.zeros(n_a)
    for ia in range(n_a):
        ie_pk = int(np.argmax(chi[ia]))
        # Quadratic vertex of three points around the peak.
        if 0 < ie_pk < n_e - 1:
            x = etas[ie_pk - 1: ie_pk + 2]
            y = chi[ia, ie_pk - 1: ie_pk + 2]
            denom = (y[0] - 2 * y[1] + y[2])
            if abs(denom) > 1e-12:
                shift = 0.5 * (y[0] - y[2]) / denom
                eta_c_chi[ia] = float(x[1] + shift * (x[1] - x[0]))
            else:
                eta_c_chi[ia] = float(etas[ie_pk])
        else:
            eta_c_chi[ia] = float(etas[ie_pk])

        # Linear interpolation for phi = 1/2 crossing.
        ph = phi[ia]
        target = 0.5
        for k in range(n_e - 1):
            if (ph[k] - target) * (ph[k + 1] - target) <= 0:
                t = (target - ph[k]) / (ph[k + 1] - ph[k])
                eta_c_phi[ia] = float(etas[k] + t * (etas[k + 1] - etas[k]))
                break
        else:
            eta_c_phi[ia] = float("nan")

    np.savez_compressed(
        DATA / "phase_curve.npz",
        alphas=alphas, etas=etas, phi_mean=phi, chi=chi,
        eta_c_chi=eta_c_chi, eta_c_phi=eta_c_phi,
        params=np.array([N, L, v0, R_r, R_a, n_warm, n_meas,
                         len(seeds)], dtype=float),
    )
    print("saved:", DATA / "phase_curve.npz")
    print("eta_c via chi peak:", np.round(eta_c_chi, 3))
    print("eta_c via phi=1/2:", np.round(eta_c_phi, 3))


if __name__ == "__main__":
    main()
