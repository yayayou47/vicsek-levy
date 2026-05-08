"""
Run a single Vicsek simulation (Gaussian or Levy) and write an MP4.

Frames show all particles in a uniform blue, with normalised arrows
indicating their orientation. An inset tracks polarisation phi(t).
Background colour follows the journal style (cream).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import style
from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent / "videos"
VIDEOS.mkdir(exist_ok=True)

style.apply()


def render_frame(sim: Vicsek, phi_hist: list[float], arrow_len: float) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=110)
    p = sim.p
    u = np.cos(sim.theta)
    v = np.sin(sim.theta)
    ax.quiver(
        sim.x, sim.y, u, v,
        color=style.PARTICLE_BLUE,
        scale=1.0 / arrow_len, scale_units="xy",
        angles="xy", width=0.004, headwidth=3.5, headlength=4.0,
    )
    ax.set_xlim(0, p.L)
    ax.set_ylim(0, p.L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Vicsek  N={p.N}  L={p.L:.0f}  $\\eta$={p.eta:.2f}  "
        f"$\\alpha$={p.alpha:.2f}   t={sim.t}"
    )
    ax2 = fig.add_axes([0.65, 0.12, 0.30, 0.18], facecolor=style.CREAM)
    ax2.plot(phi_hist, color="black", lw=1.0)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, max(50, len(phi_hist)))
    ax2.set_title(r"$\varphi(t)$", fontsize=9)
    ax2.tick_params(labelsize=7)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--L", type=float, default=32.0)
    ap.add_argument("--v0", type=float, default=0.05)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--arrow-len", type=float, default=0.6)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    p = VicsekParams(
        N=args.N, L=args.L, v0=args.v0,
        eta=args.eta, alpha=args.alpha, seed=args.seed,
    )
    sim = Vicsek(p)
    out = args.out or str(
        VIDEOS / f"vicsek_alpha{args.alpha:.2f}_eta{args.eta:.2f}.mp4"
    )

    phi_hist = []
    writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=7)
    try:
        for k in tqdm(range(args.steps), desc="movie"):
            sim.step()
            phi_hist.append(sim.polarisation())
            if k % args.every == 0:
                writer.append_data(render_frame(sim, phi_hist, args.arrow_len))
    finally:
        writer.close()
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
