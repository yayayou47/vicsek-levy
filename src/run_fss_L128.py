"""
Phase-2 FSS endpoint at L=128. Thin wrapper around fss_runner.

Output: data/fss_L128.npz

Compute envelope (8 workers): ~10 h wallclock at warm=30000.
The script writes a partial fss_L128.partial.npz every
--checkpoint-every (default 30) completed jobs and can resume from
it on relaunch.

Launch:
    .venv/bin/python version1/src/run_fss_L128.py
"""
from __future__ import annotations

import argparse

from fss_runner import FSSConfig, run_fss_sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-warm", type=int, default=30000)
    parser.add_argument("--n-meas", type=int, default=1500)
    parser.add_argument("--checkpoint-every", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = FSSConfig(L=128.0, n_warm=args.n_warm, n_meas=args.n_meas)
    run_fss_sweep(
        cfg, "fss_L128.npz",
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
