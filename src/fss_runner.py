"""
Unified FSS-sweep runner. Factors the boilerplate that the
run_fss_L*.py scripts used to duplicate: parameter setup, the
ProcessPoolExecutor.map loop, the (alpha, eta, seed) -> index
bookkeeping, the partial-checkpoint / resume protocol, and the
np.savez_compressed schema.

Usage from a thin driver:

    from fss_runner import FSSConfig, run_fss_sweep

    cfg = FSSConfig(L=91.0, sigma=500.0/15.0**2,
                    n_warm=20000, n_meas=1500)
    run_fss_sweep(cfg, "fss_L91.npz",
                  workers=8, checkpoint_every=30, resume=True)

The schema written by run_fss_sweep matches what
refit_fss_M2.py and update_chiffres.py expect:
    L (scalar), N (scalar), alphas, etas, seeds,
    phi[a, e, s], chi[a, e, s], U4[a, e, s],
    sigma (scalar), params = [v0, n_warm, n_meas]
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from analysis import time_series_stats
from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)

DEFAULT_ETAS = (
    0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
    0.24, 0.30,
)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_ALPHAS = (1.0, 2.0)


@dataclass
class FSSConfig:
    """Operating-point and protocol parameters for a single-L sweep.

    Defaults match the no-blind two-zone Vicsek operating point
    identified by the noblind R_r scan at L=22: the chi-peak is
    sharpest at R_r=0.45, sigma=2.22, with full 360 deg vision
    (blind sector removed from the model in 2026-05)."""
    L: float
    sigma: float = 500.0 / 15.0**2       # baseline density of the manuscript
    v0: float = 0.05
    R_r: float = 0.45
    R_a: float = 0.7
    n_warm: int = 20000
    n_meas: int = 1500
    alphas: tuple = DEFAULT_ALPHAS
    etas: tuple = DEFAULT_ETAS
    seeds: tuple = DEFAULT_SEEDS

    @property
    def N(self) -> int:
        return int(round(self.sigma * self.L * self.L))

    def jobs(self) -> list[tuple]:
        return [
            (self.N, float(self.L), self.v0, self.R_r, self.R_a,
             float(eta), float(alpha), int(sd),
             self.n_warm, self.n_meas)
            for alpha in self.alphas
            for eta in self.etas
            for sd in self.seeds
        ]


def _run_one(args):
    """Worker: simulate a single (alpha, eta, seed) point and return
    summary stats. Module-level so the ProcessPoolExecutor can pickle
    it across workers."""
    N, L, v0, R_r, R_a, eta, alpha, seed, n_warm, n_meas = args
    p = VicsekParams(
        N=int(N), L=float(L), v0=v0,
        R_r=R_r, R_a=R_a,
        eta=float(eta), alpha=float(alpha), seed=int(seed),
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
    return alpha, eta, seed, (s["phi_mean"], s["chi"], s["binder"])


def _save(out_path: Path, cfg: FSSConfig,
          phi, chi, U4, n_done: int = -1) -> None:
    """Write the canonical schema. n_done >= 0 marks a partial."""
    payload = dict(
        L=np.array(float(cfg.L)),
        N=np.array(cfg.N),
        alphas=np.array(cfg.alphas),
        etas=np.array(cfg.etas),
        seeds=np.array(cfg.seeds),
        phi=phi, chi=chi, U4=U4,
        sigma=np.array(cfg.sigma),
        params=np.array([cfg.v0, cfg.n_warm, cfg.n_meas], dtype=float),
    )
    if n_done >= 0:
        payload["n_done"] = np.array(n_done, dtype=np.int64)
    np.savez_compressed(out_path, **payload)


def run_fss_sweep(
    cfg: FSSConfig,
    output_name: str,
    *,
    workers: int = 8,
    checkpoint_every: int = 30,
    resume: bool = False,
    desc: str | None = None,
) -> Path:
    """Run a full FSS sweep at one L, write the canonical .npz under
    data/<output_name>. Returns the output path. Use --resume from a
    matching .partial.npz to skip already-completed jobs."""
    out_path = DATA / output_name
    partial_path = out_path.with_suffix(".partial.npz")

    n_a, n_e, n_s = (len(cfg.alphas), len(cfg.etas), len(cfg.seeds))
    phi = np.zeros((n_a, n_e, n_s))
    chi = np.zeros_like(phi)
    U4 = np.zeros_like(phi)

    seed_index = {int(s): i for i, s in enumerate(cfg.seeds)}
    alpha_index = {float(a): i for i, a in enumerate(cfg.alphas)}
    eta_index = {float(e): i for i, e in enumerate(cfg.etas)}

    skip_first = 0
    if resume and partial_path.exists():
        z_p = np.load(partial_path)
        prev_warm = int(z_p["params"][1])
        prev_meas = int(z_p["params"][2])
        if prev_warm == cfg.n_warm and prev_meas == cfg.n_meas:
            phi = z_p["phi"].copy()
            chi = z_p["chi"].copy()
            U4 = z_p["U4"].copy()
            skip_first = int(z_p.get("n_done", 0))
            print(f"[resume] {skip_first}/{n_a * n_e * n_s} jobs "
                  f"already in partial; replaying remainder.")
        else:
            print(f"[resume] partial has warm={prev_warm}, "
                  f"meas={prev_meas}; current has warm={cfg.n_warm}, "
                  f"meas={cfg.n_meas}; ignoring partial.")

    jobs_all = cfg.jobs()
    jobs_to_do = jobs_all[skip_first:]
    n_done = skip_first
    n_total = len(jobs_all)

    print(f"L = {cfg.L}, N = {cfg.N}, sigma = {cfg.sigma:.4f}")
    print(f"jobs = {n_total} (running {len(jobs_to_do)}), "
          f"workers = {workers}, warm = {cfg.n_warm}, "
          f"meas = {cfg.n_meas}")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(ex.map(_run_one, jobs_to_do),
                      total=len(jobs_to_do),
                      desc=desc or f"fss_L{int(cfg.L)}"):
            alpha, eta, seed, (ph, ch, bi) = r
            ia = alpha_index[float(alpha)]
            ie = eta_index[float(eta)]
            isd = seed_index[int(seed)]
            phi[ia, ie, isd] = ph
            chi[ia, ie, isd] = ch
            U4[ia, ie, isd] = bi
            n_done += 1
            if (n_done % checkpoint_every == 0
                    and n_done < n_total):
                _save(partial_path, cfg, phi, chi, U4, n_done=n_done)
    dt = time.time() - t0

    _save(out_path, cfg, phi, chi, U4)
    if partial_path.exists():
        partial_path.unlink()
    print(f"saved: {out_path}    "
          f"[{dt/60:.1f} min on {workers} workers]")
    return out_path
