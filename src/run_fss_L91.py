"""
Intermediate FSS point at L=91. Thin wrapper around fss_runner.

Output: data/fss_L91.npz

Wall-clock on 8 workers: ~3 h at warm=20000.

Launch:
    .venv/bin/python version1/src/run_fss_L91.py
"""
from __future__ import annotations

import argparse

from fss_runner import FSSConfig, run_fss_sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-warm", type=int, default=20000)
    parser.add_argument("--n-meas", type=int, default=1500)
    parser.add_argument("--checkpoint-every", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = FSSConfig(L=91.0, n_warm=args.n_warm, n_meas=args.n_meas)
    run_fss_sweep(
        cfg, "fss_L91.npz",
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
