"""
Phase diagram with noise calibration across alpha.

The bare scale parameter eta is not the angular disorder actually
injected into the dynamics. After wrapping to (-pi, pi], the relevant
quantity is the circular variance

   V(alpha, eta) = 1 - |E[e^{i xi}]|,   xi ~ S_alpha(scale=eta, beta=0).

V grows monotonically in eta from 0 to (1 - 1/e) = 0.632 (uniform).
For each alpha we (i) tabulate V(alpha, eta) on a fine eta grid by
Monte Carlo, (ii) invert it to obtain eta_alpha(V), and (iii) sweep V
in [0, 0.6]. This calibrates the angular disorder so that any shift of
the transition reflects the noise *shape*, not its amplitude.

Output: data/calibrated_sweep.npz
   alphas, V_grid, eta_table[a, v] (calibrated noise scales),
   phi_mean[a, v], chi[a, v], binder[a, v]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from noise import stable_rvs
from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def circular_variance(alpha: float, eta: float, n: int,
                      rng: np.random.Generator) -> float:
    if eta == 0.0:
        return 0.0
    xi = stable_rvs(alpha, eta, n, rng)
    xi = (xi + np.pi) % (2 * np.pi) - np.pi
    R = np.hypot(np.mean(np.cos(xi)), np.mean(np.sin(xi)))
    return float(1.0 - R)


def calibrate(alpha: float, V_target: np.ndarray, n_mc: int = 200_000,
              eta_max: float = 6.0,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Return eta_alpha(V) by inverting V(eta) on a fine grid."""
    if rng is None:
        rng = np.random.default_rng(0)
    eta_grid = np.concatenate(([0.0], np.geomspace(1e-3, eta_max, 200)))
    V_grid = np.array(
        [circular_variance(alpha, e, n_mc, rng) for e in eta_grid]
    )
    # V is non-monotonic in MC noise; smooth by enforcing monotonicity.
    V_grid = np.maximum.accumulate(V_grid)
    return np.interp(V_target, V_grid, eta_grid)


def measure(p: VicsekParams, n_warm: int, n_meas: int) -> tuple:
    sim = Vicsek(p)
    sim.theta[:] = 0.0
    for _ in range(n_warm):
        sim.step()
    phis = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        phis[k] = sim.polarisation()
    return phis.mean(), p.N * phis.var(), 1.0 - np.mean(phis**4) / (
        3.0 * np.mean(phis**2) ** 2
    )


def main():
    N, L, v0 = 500, 15.0, 0.05
    R_r, R_a = 0.45, 0.7
    n_warm, n_meas = 1500, 1500
    seeds = [11, 23, 41]

    alphas = np.array([1.0, 1.5, 2.0])
    V_grid = np.linspace(0.0, 0.6, 16)
    n_a, n_v = len(alphas), len(V_grid)

    # Step 1: calibrate eta_alpha(V) for each alpha.
    eta_table = np.zeros((n_a, n_v))
    rng = np.random.default_rng(0)
    print("calibrating noise scales ...")
    for ia, alpha in enumerate(alphas):
        eta_table[ia] = calibrate(float(alpha), V_grid, rng=rng)

    # Step 2: measure phi, chi, U_4 at each (alpha, V).
    phi = np.zeros((n_a, n_v))
    chi = np.zeros((n_a, n_v))
    binder = np.zeros((n_a, n_v))

    pbar = tqdm(total=n_a * n_v * len(seeds), desc="calibrated")
    for ia, alpha in enumerate(alphas):
        for iv in range(n_v):
            eta_eff = float(eta_table[ia, iv])
            ph_acc = 0.0
            ch_acc = 0.0
            bi_acc = 0.0
            for seed in seeds:
                p = VicsekParams(
                    N=N, L=L, v0=v0, R_r=R_r, R_a=R_a,
                    eta=eta_eff, alpha=float(alpha), seed=seed,
                )
                ph, ch, bi = measure(p, n_warm, n_meas)
                ph_acc += ph
                ch_acc += ch
                bi_acc += bi
                pbar.update(1)
            phi[ia, iv] = ph_acc / len(seeds)
            chi[ia, iv] = ch_acc / len(seeds)
            binder[ia, iv] = bi_acc / len(seeds)
    pbar.close()

    np.savez_compressed(
        DATA / "calibrated_sweep.npz",
        alphas=alphas, V_grid=V_grid, eta_table=eta_table,
        phi_mean=phi, chi=chi, binder=binder,
        params=np.array([N, L, v0, R_r, R_a, n_warm, n_meas,
                         len(seeds)], dtype=float),
    )
    print("saved:", DATA / "calibrated_sweep.npz")


if __name__ == "__main__":
    main()
