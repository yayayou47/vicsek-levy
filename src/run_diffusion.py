"""
Angular and spatial diffusion in the zonal Vicsek-Couzin model.

We track unfolded particle trajectories (x, y) -- accumulating periodic
jumps -- and the unwrapped heading angle theta, then measure

   MSD_theta(t) = < (theta_i(t) - theta_i(0))^2 >_i,seeds
   MSD_x(t)     = < |Delta r_i(t)|^2 >_i,seeds

For Gaussian noise (alpha = 2) MSD_theta should grow linearly in t in
the disordered phase. Heavy-tailed noise (alpha < 2) injects
S_alpha-distributed angular increments with infinite variance, and
the empirical second moment of theta grows faster than t -- the
hallmark of angular super-diffusion at small alpha.

Output: data/diffusion.npz
   alphas, etas, t, msd_theta[a, e, k], msd_x[a, e, k]
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
    R_r, R_a = 0.5, 0.7
    n_warm = 1500
    n_meas = 2000
    n_log = 60
    seeds = [11, 23, 41]

    alphas = np.array([1.0, 1.5, 2.0])
    etas = np.array([0.05, 0.15, 0.30])

    t_samples = np.unique(
        np.round(np.geomspace(1, n_meas, n_log)).astype(int)
    )
    n_t = len(t_samples)

    msd_theta = np.zeros((len(alphas), len(etas), n_t))
    msd_x = np.zeros((len(alphas), len(etas), n_t))

    pbar = tqdm(total=len(alphas) * len(etas) * len(seeds),
                desc="diffusion")
    for ia, alpha in enumerate(alphas):
        for ie, eta in enumerate(etas):
            for seed in seeds:
                p = VicsekParams(
                    N=N, L=L, v0=v0, R_r=R_r, R_a=R_a,
                    eta=float(eta), alpha=float(alpha), seed=seed,
                )
                sim = Vicsek(p)
                sim.theta[:] = 0.0
                for _ in range(n_warm):
                    sim.step()

                theta_ref = sim.theta.copy()
                x_ref = sim.x.copy()
                y_ref = sim.y.copy()

                theta_unwrap = sim.theta.copy()
                theta_prev = sim.theta.copy()
                x_unfold = sim.x.copy()
                y_unfold = sim.y.copy()
                x_prev = sim.x.copy()
                y_prev = sim.y.copy()

                idx_sample = 0
                for k in range(1, n_meas + 1):
                    sim.step()

                    dtheta = sim.theta - theta_prev
                    dtheta -= 2 * np.pi * np.round(dtheta / (2 * np.pi))
                    theta_unwrap += dtheta
                    theta_prev = sim.theta.copy()

                    dx = sim.x - x_prev
                    dy = sim.y - y_prev
                    dx -= L * np.round(dx / L)
                    dy -= L * np.round(dy / L)
                    x_unfold += dx
                    y_unfold += dy
                    x_prev = sim.x.copy()
                    y_prev = sim.y.copy()

                    if idx_sample < n_t and k == t_samples[idx_sample]:
                        dth = theta_unwrap - theta_ref
                        msd_theta[ia, ie, idx_sample] += np.mean(dth ** 2)
                        rx = x_unfold - x_ref
                        ry = y_unfold - y_ref
                        msd_x[ia, ie, idx_sample] += np.mean(rx**2 + ry**2)
                        idx_sample += 1
                pbar.update(1)
    pbar.close()

    msd_theta /= len(seeds)
    msd_x /= len(seeds)

    np.savez_compressed(
        DATA / "diffusion.npz",
        alphas=alphas, etas=etas, t=t_samples,
        msd_theta=msd_theta, msd_x=msd_x,
        params=np.array([N, L, v0, R_r, R_a, n_warm, n_meas,
                         len(seeds)], dtype=float),
    )
    print("saved:", DATA / "diffusion.npz")


if __name__ == "__main__":
    main()
