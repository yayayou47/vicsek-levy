"""
Slow eta-ramp hysteresis test. For each alpha, ramp eta from 0 to
eta_max linearly over T_up steps, then back down to 0 over T_down.
Record the instantaneous polarisation phi(t) and noise eta(t). A
hysteresis loop is the canonical signature of a first-order transition;
its absence under heavy tails would confirm the FSS conclusion.

Output: data/hysteresis.npz
   alphas, eta_traj[a, t], phi_traj[a, t], direction[t]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def main():
    N, L, v0 = 500, 15.0, 0.05
    R_r, R_a = 0.45, 0.7
    eta_max = 0.40
    T_relax = 800
    T_up = 8000
    T_down = 8000
    alphas = np.array([1.0, 1.5, 2.0])
    seeds = [17, 23, 41]

    n_steps = T_up + T_down
    eta_up = np.linspace(0.0, eta_max, T_up, endpoint=False)
    eta_down = np.linspace(eta_max, 0.0, T_down, endpoint=True)
    eta_path = np.concatenate([eta_up, eta_down])
    direction = np.concatenate([np.ones(T_up), -np.ones(T_down)])

    phi_traj = np.zeros((len(alphas), n_steps))
    pbar = tqdm(total=len(alphas) * len(seeds) * n_steps,
                desc="hysteresis")
    for ia, alpha in enumerate(alphas):
        accum = np.zeros(n_steps)
        for seed in seeds:
            p = VicsekParams(
                N=N, L=L, v0=v0, R_r=R_r, R_a=R_a,
                eta=0.0, alpha=float(alpha), seed=seed,
            )
            sim = Vicsek(p)
            sim.theta[:] = 0.0
            for _ in range(T_relax):
                sim.step()
            for k, eta_k in enumerate(eta_path):
                sim.p = VicsekParams(
                    N=N, L=L, v0=v0, R_r=R_r, R_a=R_a,
                    eta=float(eta_k), alpha=float(alpha), seed=seed,
                )
                sim.step()
                accum[k] += sim.polarisation()
                pbar.update(1)
        phi_traj[ia] = accum / len(seeds)
    pbar.close()

    np.savez_compressed(
        DATA / "hysteresis.npz",
        alphas=alphas,
        eta_path=eta_path,
        direction=direction,
        phi_traj=phi_traj,
        params=np.array([N, L, v0, R_r, R_a, T_relax, T_up, T_down,
                         len(seeds)], dtype=float),
    )
    print("saved:", DATA / "hysteresis.npz")


if __name__ == "__main__":
    main()
