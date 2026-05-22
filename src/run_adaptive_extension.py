"""
Extend the density-adaptive FSS pilot from L in {15, 22, 30, 45} to
the harmonised 7-size lever L in {15, 22, 30, 45, 64, 91, 128}.
Writes data/adaptive_perseed_ext.npz with shape consistent with the
existing adaptive_perseed.npz so the make_figures builder can
concatenate the two via Ls.

Output: data/adaptive_perseed_ext.npz with the three new sizes only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
V2_SRC = HERE.parent.parent.parent / "version2" / "src" \
         if HERE.parent.name == "notes" else \
         HERE.parent.parent / "version2" / "src"
sys.path.insert(0, str(V2_SRC))

from vicsek_adaptive import AdaptiveParams, AdaptiveVicsek  # noqa: E402

DATA = HERE.parent / "data" if HERE.parent.name != "notes" else \
       HERE.parent.parent / "data"


def measure(p, n_warm, n_meas, track=False):
    sim = AdaptiveVicsek(p)
    sim.theta[:] = 0.0
    for _ in range(n_warm):
        sim.step()
    phi = np.empty(n_meas)
    a_acc = []
    for k in range(n_meas):
        sim.step()
        phi[k] = sim.polarisation()
        if track:
            a_acc.append(sim.alpha_i.copy())
    phi_mean = float(phi.mean())
    chi = p.N * float(phi.var())
    binder = 1.0 - float(np.mean(phi**4) / (3.0 * np.mean(phi**2) ** 2))
    if track:
        amat = np.stack(a_acc, axis=0)
        am = float(amat.mean())
        as_ = float(amat.std(axis=1).mean())
    else:
        am, as_ = float('nan'), float('nan')
    return phi_mean, chi, binder, (am, as_)


def main():
    sigma = 2.22
    Ls = np.array([64.0, 91.0, 128.0])
    Ns = (sigma * Ls**2).round().astype(int)
    v0 = 0.05
    R_r, R_a = 0.45, 0.7
    n_warm, n_meas = 1500, 1000
    seeds = np.array([0, 1, 2, 3, 4])
    etas = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50])
    modes = [("fixed", 1.0, 1.0), ("adaptive", 1.0, 2.0)]

    n_m, n_L, n_e, n_s = len(modes), len(Ls), len(etas), len(seeds)
    phi = np.zeros((n_m, n_L, n_e, n_s))
    chi = np.zeros_like(phi)
    U4 = np.zeros_like(phi)
    am = np.full((n_m, n_L, n_e, n_s), np.nan)
    as_ = np.full((n_m, n_L, n_e, n_s), np.nan)

    pbar = tqdm(total=n_m * n_L * n_e * n_s, desc="adaptive_ext")
    for im, (label, a_min, a_max) in enumerate(modes):
        track = (label == "adaptive")
        for iL, (L, N) in enumerate(zip(Ls, Ns)):
            for ie, eta in enumerate(etas):
                for isd, sd in enumerate(seeds):
                    p = AdaptiveParams(
                        N=int(N), L=float(L), v0=v0, R_r=R_r, R_a=R_a,
                        eta=float(eta),
                        alpha_min=a_min, alpha_max=a_max,
                        n_star=3.0, slope=2.0, seed=int(sd),
                    )
                    ph, ch, bi, (a_m, a_s) = measure(
                        p, n_warm, n_meas, track=track)
                    phi[im, iL, ie, isd] = ph
                    chi[im, iL, ie, isd] = ch
                    U4[im, iL, ie, isd] = bi
                    am[im, iL, ie, isd] = a_m
                    as_[im, iL, ie, isd] = a_s
                    pbar.update(1)
    pbar.close()

    out = DATA / "adaptive_perseed_ext.npz"
    np.savez_compressed(
        out,
        modes=np.array([m[0] for m in modes]),
        Ls=Ls, Ns=Ns, etas=etas, seeds=seeds,
        phi=phi, chi=chi, U4=U4,
        alpha_mean=am, alpha_std=as_,
        params=np.array([v0, R_r, R_a, n_warm, n_meas], dtype=float),
    )
    print("saved:", out)


if __name__ == "__main__":
    main()
