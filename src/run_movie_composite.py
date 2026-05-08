"""
Single composite Supplementary movie that contrasts the four regimes
discussed in the manuscript:
  (a) Gaussian polar       alpha = 2.0, eta = 0.05  (band soliton)
  (b) Gaussian near-FSS    alpha = 2.0, eta = 0.15  (intermittent)
  (c) Cauchy near-critical alpha = 1.0, eta = 0.05  (band erosion)
  (d) Cauchy disordered    alpha = 1.0, eta = 0.20  (gas)

Particles are coloured by their heading angle through a cyclic HSV
colormap, so an ordered phase reads as a uniform colour while a
disordered phase reads as a rainbow. A bottom strip carries the
synchronous polarisation traces phi(t) of all four panels.

Output: videos/vicsek_composite.mp4
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from tqdm import tqdm

import style
from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent / "videos"
VIDEOS.mkdir(exist_ok=True)
style.apply()


REGIMES = [
    ("(a) Gaussian polar (band)",      2.0, 0.05),
    ("(b) Gaussian near-FSS",          2.0, 0.15),
    ("(c) Cauchy near-critical",       1.0, 0.05),
    ("(d) Cauchy disordered",          1.0, 0.20),
]
N, L, v0 = 2000, 30.0, 0.05
R_r, R_a = 0.5, 0.7
N_WARM = 1500
N_FRAMES = 400
SUBSAMPLE = 2          # one rendered frame per SUBSAMPLE simulation steps
FPS = 24
ARROW_LEN = 0.6


def heading_to_rgb(theta: np.ndarray) -> np.ndarray:
    """Map heading in (-pi, pi] to a cyclic HSV colormap (full saturation,
    high value)."""
    h = (theta + np.pi) / (2.0 * np.pi)
    s = np.full_like(h, 0.85)
    v = np.full_like(h, 0.95)
    hsv = np.stack([h, s, v], axis=-1)
    return hsv_to_rgb(hsv)


def build_sims():
    sims = []
    for label, alpha, eta in REGIMES:
        p = VicsekParams(
            N=N, L=L, v0=v0, R_r=R_r, R_a=R_a,
            eta=float(eta), alpha=float(alpha), seed=0,
        )
        sim = Vicsek(p)
        sim.theta[:] = 0.0
        sims.append(sim)
    return sims


def warm(sims):
    for sim in tqdm(sims, desc="warm-up"):
        for _ in range(N_WARM):
            sim.step()


def render_frame(sims, phi_hist, t):
    fig = plt.figure(figsize=(10.0, 7.0), dpi=110, facecolor=style.CREAM)
    gs = fig.add_gridspec(3, 4, height_ratios=[3.0, 3.0, 1.4],
                          hspace=0.25, wspace=0.18,
                          left=0.04, right=0.98, top=0.94, bottom=0.07)

    panel_axes = []
    for i, (label, alpha, eta) in enumerate(REGIMES):
        r, c = divmod(i, 2)
        ax = fig.add_subplot(gs[r, 2 * c:2 * (c + 1)])
        sim = sims[i]
        u = np.cos(sim.theta); v = np.sin(sim.theta)
        rgb = heading_to_rgb(sim.theta)
        ax.quiver(
            sim.x, sim.y, u, v, color=rgb,
            scale=1.0 / ARROW_LEN, scale_units="xy",
            angles="xy", width=0.004, headwidth=3.5, headlength=4.0,
        )
        ax.set_xlim(0, L); ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        phi_now = phi_hist[i][-1] if phi_hist[i] else 0.0
        ax.set_title(
            f"{label}    "
            fr"$\alpha\,$=$\,${alpha:.1f},  $\eta\,$=$\,${eta:.2f},   "
            fr"$\langle\varphi\rangle\,$=$\,${phi_now:.2f}",
            fontsize=10, pad=4,
        )
        panel_axes.append(ax)

    # Bottom strip: synchronous phi(t) traces
    ax_phi = fig.add_subplot(gs[2, :])
    cols = ["#1f4ea1", "#5a7eb4", "#d76f3a", "#a23a1f"]
    labels = ["(a)", "(b)", "(c)", "(d)"]
    for i, ph in enumerate(phi_hist):
        ax_phi.plot(np.arange(len(ph)) * SUBSAMPLE, ph,
                    color=cols[i], lw=1.3, label=labels[i])
    ax_phi.set_xlim(0, N_FRAMES * SUBSAMPLE)
    ax_phi.set_ylim(0, 1.05)
    ax_phi.set_xlabel("simulation step (post warm-up)")
    ax_phi.set_ylabel(r"polarisation $\varphi$")
    ax_phi.legend(loc="upper right", ncol=4, fontsize=9, framealpha=0.9)
    ax_phi.tick_params(labelsize=8)
    ax_phi.set_title(
        f"step {t}    /    Vicsek-Couzin, $L = {int(L)}$, $N = {N}$, "
        rf"$\sigma = N/L^2 \simeq 2.22$",
        fontsize=10, loc="left",
    )

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main():
    sims = build_sims()
    print(f"warm-up {N_WARM} steps for {len(sims)} regimes")
    warm(sims)

    phi_hist = [[] for _ in sims]
    out = VIDEOS / "vicsek_composite.mp4"
    writer = imageio.get_writer(str(out), fps=FPS,
                                codec="libx264", quality=8)
    try:
        for k in tqdm(range(N_FRAMES * SUBSAMPLE), desc="record"):
            for i, sim in enumerate(sims):
                sim.step()
            if k % SUBSAMPLE == 0:
                for i, sim in enumerate(sims):
                    phi_hist[i].append(sim.polarisation())
                writer.append_data(render_frame(sims, phi_hist, k))
    finally:
        writer.close()
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
