"""
FSS extension to L=64 for alpha in {1, 2} on a refined eta grid that
brackets the FSS-located eta_c. Thin wrapper around fss_runner.

Output: data/fss_L64.npz

The canonical L=64 dataset of the manuscript was produced with
warm=2000, meas=1200 (the values kept as defaults here). For a
run consistent with the L=91/L=128 protocol, override via
--n-warm 20000 (writes the warm-tagged file
data/fss_L64.warm20000.npz via run_fss_smallL.py instead).

Launch:
    .venv/bin/python version1/src/run_fss_L64.py
"""
from __future__ import annotations

import argparse

from fss_runner import FSSConfig, run_fss_sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-warm", type=int, default=2000)
    parser.add_argument("--n-meas", type=int, default=1200)
    parser.add_argument("--checkpoint-every", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = FSSConfig(L=64.0, n_warm=args.n_warm, n_meas=args.n_meas)
    run_fss_sweep(
        cfg, "fss_L64.npz",
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
