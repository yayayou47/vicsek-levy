"""
Giant Number Fluctuations (GNF) scaling. We measure
Var(N_box) vs <N_box> in nested sub-boxes for ordered-phase
configurations at three values of the Levy index alpha. A linear fit
in log-log space gives the exponent zeta in
   Var(N_box) ~ <N_box>^(2*zeta).
zeta = 1/2 is the equilibrium / Poisson value; the polar flocking
phase exhibits zeta ~ 0.8 (Toner-Tu).

Output: data/gnf.npz with arrays
   alphas, etas, means[i, k], vars[i, k]
where i indexes the (alpha, eta) case and k indexes the box-size bin.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from analysis import density_fluctuations
from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def main():
    N, L, v0 = 2000, 30.0, 0.05            # density sigma = 2.22
    R_r, R_a = 0.5, 0.7
    n_warm = 2000
    n_meas = 6000
    n_skip = 50
    n_box_sizes = 12
    seeds = [11, 23, 41]

    # Three ordered-phase operating points, one per alpha. eta chosen
    # so that <phi> ~ 0.9 in each case.
    cases = [(2.0, 0.10), (1.5, 0.05), (1.0, 0.02)]
    means = np.zeros((len(cases), n_box_sizes))
    vars_ = np.zeros((len(cases), n_box_sizes))

    pbar = tqdm(total=len(cases) * len(seeds), desc="gnf")
    for ic, (alpha, eta) in enumerate(cases):
        sum_m = np.zeros(n_box_sizes)
        sum_v = np.zeros(n_box_sizes)
        n_acc = 0
        for seed in seeds:
            p = VicsekParams(
                N=N, L=L, v0=v0, R_r=R_r, R_a=R_a,
                eta=float(eta), alpha=float(alpha), seed=seed,
            )
            sim = Vicsek(p)
            sim.theta[:] = 0.0
            for _ in range(n_warm):
                sim.step()
            for k in range(n_meas):
                sim.step()
                if k % n_skip == 0:
                    m, v = density_fluctuations(sim.x, sim.y, L,
                                                n_box_sizes)
                    sum_m += m
                    sum_v += v
                    n_acc += 1
            pbar.update(1)
        means[ic] = sum_m / n_acc
        vars_[ic] = sum_v / n_acc
    pbar.close()

    np.savez_compressed(
        DATA / "gnf.npz",
        alphas=np.array([c[0] for c in cases]),
        etas=np.array([c[1] for c in cases]),
        means=means, vars=vars_,
        params=np.array([N, L, v0, R_r, R_a,
                         n_warm, n_meas, n_skip,
                         n_box_sizes, len(seeds)], dtype=float),
    )
    print("saved:", DATA / "gnf.npz")


if __name__ == "__main__":
    main()
