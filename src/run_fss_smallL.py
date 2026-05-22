"""
Small-L FSS re-run at L in {15, 22, 30, 45, 64} with the long
warm-up that matches the L=91 and L=128 protocols. Thin wrapper
around fss_runner.

Output: data/fss_L{N}.warm{n_warm}.npz per L.

Wall-clock estimate on 8 workers at warm=20000: ~100 min total
(L=15..45 are minutes each, L=64 is ~50 min).

Launch:
    .venv/bin/python version1/src/run_fss_smallL.py
"""
from __future__ import annotations

import argparse
import time

from fss_runner import FSSConfig, run_fss_sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-warm", type=int, default=20000,
                        help="Match L=91 protocol (default 20000)")
    parser.add_argument("--n-meas", type=int, default=1500)
    parser.add_argument("--Ls", type=int, nargs="+",
                        default=[15, 22, 30, 45, 64])
    args = parser.parse_args()

    print(f"Re-running FSS at L = {args.Ls} with warm = {args.n_warm}")
    t0 = time.time()
    for L in args.Ls:
        cfg = FSSConfig(L=float(L), n_warm=args.n_warm,
                        n_meas=args.n_meas)
        run_fss_sweep(
            cfg, f"fss_L{L}.warm{args.n_warm}.npz",
            workers=args.workers,
            desc=f"fss_L{L}",
        )
    print(f"\n[done] total wallclock: "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
