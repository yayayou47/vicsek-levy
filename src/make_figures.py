"""
Read data/phase_sweep.npz and produce publication figures in figures/.

  fig_order_param.pdf  : phi(eta) for each alpha
  fig_susceptibility.pdf : chi(eta) for each alpha
  fig_binder.pdf       : Binder cumulant U4(eta) for each alpha
  fig_noise_pdf.pdf    : example pdf of S_alpha noise (visualises tails)
  fig_snapshots.pdf    : 3-regime particle snapshots (ordered / critical / disordered)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style
from noise import stable_rvs
from vicsek import Vicsek, VicsekParams


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)


style.apply()


def _save(fig, name: str):
    out = FIG / name
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"saved: {out}")


def fig_phase(npz_path: Path):
    z = np.load(npz_path)
    alphas = z["alphas"]
    etas = z["etas"]

    fig, ax = plt.subplots(figsize=style.SINGLE_COL)
    for ia, a in enumerate(alphas):
        ax.plot(etas, z["phi_mean"][ia], "o-", label=fr"$\alpha={a:.2f}$")
    ax.set_xlabel(r"noise scale $\eta$")
    ax.set_ylabel(r"polarisation $\langle\varphi\rangle$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig_order_param.pdf")

    fig, ax = plt.subplots(figsize=style.SINGLE_COL)
    for ia, a in enumerate(alphas):
        ax.plot(etas, z["chi"][ia], "s-", label=fr"$\alpha={a:.2f}$")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\chi = N\,\mathrm{Var}(\varphi)$")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig_susceptibility.pdf")

    fig, ax = plt.subplots(figsize=style.SINGLE_COL)
    for ia, a in enumerate(alphas):
        ax.plot(etas, z["binder"][ia], "d-", label=fr"$\alpha={a:.2f}$")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"Binder cumulant $U_4$")
    ax.axhline(2.0 / 3.0, ls=":", c="grey", lw=0.8)
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig_binder.pdf")


def fig_noise_pdf():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=style.SINGLE_COL)
    bins = np.linspace(-6, 6, 200)
    centers = 0.5 * (bins[1:] + bins[:-1])
    for a, ls in zip([2.0, 1.5, 1.0, 0.7], ["-", "--", "-.", ":"]):
        s = stable_rvs(a, 1.0, 200_000, rng)
        s = s[(s > -6) & (s < 6)]
        h, _ = np.histogram(s, bins=bins, density=True)
        ax.semilogy(centers, h + 1e-6, ls, lw=1.3, label=fr"$\alpha={a}$")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"pdf $p_\alpha(\xi)$")
    ax.set_ylim(1e-4, 1.0)
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig_noise_pdf.pdf")


def _run_for_snapshot(alpha: float, eta: float, N: int, L: float,
                      n_warm: int, n_meas: int, seed: int):
    p = VicsekParams(N=N, L=L, v0=0.05, eta=eta, alpha=alpha, seed=seed)
    sim = Vicsek(p)
    for _ in range(n_warm):
        sim.step()
    phi_samples = np.empty(n_meas)
    for k in range(n_meas):
        sim.step()
        phi_samples[k] = sim.polarisation()
    return sim, phi_samples


def fig_fss(npz_path: Path):
    """Finite-size scaling of <phi>, chi, U4 vs eta for several L,
    one row per Levy index alpha.
    """
    z = np.load(npz_path)
    Ls = z["Ls"]
    alphas = z["alphas"]
    etas = z["etas"]
    phi = z["phi_mean"]      # (n_L, n_alpha, n_eta)
    chi = z["chi"]
    U4 = z["U4"]

    n_alpha = len(alphas)
    fig, axes = plt.subplots(n_alpha, 3,
                             figsize=(style.DOUBLE_COL[0], 4.0),
                             sharex=True)
    cmap = plt.get_cmap("plasma")
    colors = [cmap(0.15 + 0.7 * i / max(1, len(Ls) - 1))
              for i in range(len(Ls))]

    for ia, alpha_val in enumerate(alphas):
        ax_phi, ax_chi, ax_U4 = axes[ia]
        for iL, L in enumerate(Ls):
            label = fr"$L={int(L)}$" if ia == 0 else None
            ax_phi.plot(etas, phi[iL, ia], "o-", color=colors[iL],
                        ms=3, lw=1.0, label=label)
            ax_chi.plot(etas, chi[iL, ia], "s-", color=colors[iL],
                        ms=3, lw=1.0)
            ax_U4.plot(etas, U4[iL, ia], "d-", color=colors[iL],
                       ms=3, lw=1.0)

        ax_phi.set_ylim(-0.03, 1.05)
        ax_phi.set_ylabel(r"$\langle\varphi\rangle$")
        ax_chi.set_ylabel(r"$\chi = N\,\mathrm{Var}(\varphi)$")
        ax_U4.set_ylabel(r"$U_4$")
        ax_U4.axhline(2.0 / 3.0, ls=":", c="grey", lw=0.7)
        for ax in (ax_phi, ax_chi, ax_U4):
            ax.tick_params(labelsize=7)
        ax_phi.text(0.04, 0.92, fr"$\alpha={alpha_val:.1f}$",
                    transform=ax_phi.transAxes, fontsize=9,
                    fontweight="bold")

    for ax in axes[-1]:
        ax.set_xlabel(r"noise scale $\eta$")
    axes[0, 0].legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.tight_layout()
    _save(fig, "fig_fss.pdf")


def fig_clusters(npz_path: Path):
    """Cluster-size distribution P(s) at one near-critical eta per alpha
    (left panel) and largest-cluster fraction <s_max>/N vs eta (right).
    """
    z = np.load(npz_path)
    alphas = z["alphas"]
    etas = z["etas"]
    hist_bins = z["hist_bins"]
    hist = z["hist"]
    smax_frac = z["smax_frac"]
    centers = np.sqrt(hist_bins[:-1] * hist_bins[1:])
    widths = np.diff(hist_bins)

    fig, axes = plt.subplots(1, 2, figsize=(style.DOUBLE_COL[0], 2.6))
    palette = ["#3aa040", "#d76f3a", "#1f4ea1"]

    near_crit = {1.0: 0.05, 1.5: 0.10, 2.0: 0.15}

    # Panel (a): P(s) at near-critical eta.
    ax = axes[0]
    for ia, alpha_val in enumerate(alphas):
        eta_t = near_crit[float(alpha_val)]
        ie = int(np.argmin(np.abs(etas - eta_t)))
        h = hist[ia, ie]
        Ps = h / (h.sum() * widths)
        ok = (h > 0) & (centers <= 200)
        ax.loglog(centers[ok], Ps[ok], "o-",
                  color=palette[ia], ms=4, lw=1.0,
                  label=fr"$\alpha={alpha_val:.1f}, "
                        fr"\eta={etas[ie]:.2f}$")
    # Reference -tau power law for visual comparison.
    ref = np.geomspace(2.0, 80.0, 50)
    ax.loglog(ref, ref**-1.7 * 0.5, "k--", lw=0.7, alpha=0.7,
              label=r"slope $-1.7$")
    ax.set_xlabel(r"cluster size $s$")
    ax.set_ylabel(r"$P(s)$")
    ax.legend(fontsize=7)
    ax.set_title("near-critical cluster distribution", fontsize=8)
    axes[0].text(-0.18, 1.04, "(a)", transform=axes[0].transAxes,
                 fontsize=10, fontweight="bold")

    # Panel (b): <s_max>/N vs eta.
    ax = axes[1]
    for ia, alpha_val in enumerate(alphas):
        ax.plot(etas, smax_frac[ia], "o-", color=palette[ia],
                ms=4, lw=1.2,
                label=fr"$\alpha={alpha_val:.1f}$")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\langle s_{\max}\rangle / N$")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7)
    ax.set_title("largest-cluster fraction", fontsize=8)
    axes[1].text(-0.18, 1.04, "(b)", transform=axes[1].transAxes,
                 fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_clusters.pdf")


def fig_topological(npz_path: Path):
    """phi(eta), chi(eta), U4(eta) for the topological variant at
    several L per alpha, with seed-mean +/- SEM ribbons and a
    bootstrap-CI chi_max(L) scaling panel."""
    z = np.load(npz_path)
    alphas = z["alphas"]
    Ls = z["Ls"]
    etas = z["etas"]
    phi = z["phi"]
    chi = z["chi"]
    binder = z["binder"]

    # Backwards compat: data without a seed axis.
    if phi.ndim == 3:
        phi = phi[..., None]
        chi = chi[..., None]
        binder = binder[..., None]
    n_seeds = phi.shape[-1]

    palette = {1.0: "#3aa040", 2.0: "#1f4ea1"}
    n_L = len(Ls)
    cmap_a1 = plt.cm.Greens(np.linspace(0.45, 0.95, n_L))
    cmap_a2 = plt.cm.Blues(np.linspace(0.45, 0.95, n_L))
    L_colors = {1.0: cmap_a1, 2.0: cmap_a2}
    markers = ["o", "s", "D", "^", "v"]

    fig, axes_grid = plt.subplots(2, 2, figsize=(style.DOUBLE_COL[0], 5.0))
    axes = axes_grid.flatten()

    panels = [
        (axes[0], phi, r"$\langle\varphi\rangle$", "polar order"),
        (axes[1], chi, r"$\chi$", "susceptibility"),
        (axes[2], binder, r"$U_4$", "Binder cumulant"),
    ]
    for ax, arr, ylabel, title in panels:
        for ia, a in enumerate(alphas):
            for il, L in enumerate(Ls):
                m = arr[ia, il].mean(axis=-1)
                se = arr[ia, il].std(axis=-1) / max(1.0, np.sqrt(n_seeds))
                col = L_colors[float(a)][il]
                ax.plot(etas, m,
                        marker=markers[il % len(markers)], ls="-",
                        color=col, ms=3.5, lw=0.9,
                        label=fr"$\alpha={a:.0f},\,L={int(L)}$")
                ax.fill_between(etas, m - se, m + se, color=col,
                                alpha=0.18, lw=0)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8)
        if title == "polar order":
            ax.set_ylim(-0.02, 1.02)
        if title == "Binder cumulant":
            ax.axhline(2.0 / 3.0, ls=":", c="grey", lw=0.7)
    axes[0].legend(fontsize=5, loc="lower left", ncol=2)

    # Panel (d): chi_max(L) with bootstrap-CI slope.
    ax = axes[3]
    rng = np.random.default_rng(0)
    n_boot = 2000
    for ia, a in enumerate(alphas):
        chi_a = chi[ia]              # (L, eta, seed)
        chi_max_per_seed = chi_a.max(axis=1)   # (L, seed)
        c_mean = chi_max_per_seed.mean(axis=-1)
        c_se = chi_max_per_seed.std(axis=-1) / max(1.0, np.sqrt(n_seeds))
        slopes = []
        for _ in range(n_boot):
            idx = rng.integers(0, n_seeds, size=n_seeds)
            cb = chi_max_per_seed[:, idx].mean(axis=-1)
            if (cb > 0).all():
                slopes.append(np.polyfit(np.log(Ls), np.log(cb), 1)[0])
        slopes = np.asarray(slopes)
        s_mean = float(slopes.mean()); s_se = float(slopes.std())
        col = palette[float(a)]
        ax.errorbar(Ls, c_mean, yerr=c_se, fmt="o-", color=col,
                    ms=4, lw=1.1, capsize=3,
                    label=fr"$\alpha={a:.0f}$: $L^{{{s_mean:.2f}\pm{s_se:.2f}}}$")
        Lf = np.linspace(Ls[0] * 0.95, Ls[-1] * 1.05, 50)
        b = np.polyfit(np.log(Ls), np.log(c_mean), 1)
        ax.plot(Lf, np.exp(b[1]) * Lf ** b[0], "--", color=col,
                lw=0.7, alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(list(Ls))
    ax.set_xticklabels([str(int(L)) for L in Ls])
    ax.minorticks_off()
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\chi_{\max}$")
    ax.set_title("FSS of susceptibility", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")

    for k, ax in enumerate(axes):
        ax.text(-0.18, 1.04, f"({chr(97+k)})", transform=ax.transAxes,
                fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_topological.pdf")


def fig_robustness(npz_path: Path):
    """Robustness of the phi(eta) curve under v_0, beta, sigma at
    alpha = 1 and 2. Each row is one axis, left column alpha=2, right
    column alpha=1. Mean across seeds with +/- SEM ribbons.
    """
    z = np.load(npz_path)
    etas = z["etas"]
    alphas = z["alphas"]
    grids = {
        "v0": (z["v0_grid"], z["phi_v0"], r"$v_0$"),
        "beta": (z["beta_grid"], z["phi_beta"], r"$\beta$ (deg)"),
        "sigma": (z["sigma_grid"], z["phi_sigma"], r"$\sigma$"),
    }
    # Detect per-seed shape (..., n_seeds); fall back to mean-only.
    has_seed_axis = next(iter(grids.values()))[1].ndim == 4

    fig, axes = plt.subplots(3, 2, figsize=(style.DOUBLE_COL[0], 6.5),
                             sharex=True, sharey=True)
    cmap = plt.cm.viridis

    for irow, (name, (grid, phi, label)) in enumerate(grids.items()):
        for ja, alpha_val in enumerate(alphas):
            ax = axes[irow, ja]
            for ig, val in enumerate(grid):
                c = cmap(ig / max(1, len(grid) - 1))
                if has_seed_axis:
                    arr = phi[ja, ig]                     # (n_eta, n_seed)
                    m = arr.mean(axis=-1)
                    n_s = arr.shape[-1]
                    se = arr.std(axis=-1) / max(1.0, np.sqrt(n_s))
                    ax.plot(etas, m, "o-", color=c, ms=3, lw=1.0,
                            label=f"{val:g}")
                    ax.fill_between(etas, m - se, m + se, color=c,
                                    alpha=0.18, lw=0)
                else:
                    ax.plot(etas, phi[ja, ig], "o-",
                            color=c, ms=3, lw=1.0, label=f"{val:g}")
            if irow == 0:
                ax.set_title(fr"$\alpha = {alpha_val:.1f}$", fontsize=8)
            ax.set_ylim(-0.02, 1.02)
            ax.legend(title=label, fontsize=6, title_fontsize=7,
                      loc="upper right")
        axes[irow, 0].set_ylabel(r"$\langle\varphi\rangle$")

    for ax in axes[-1]:
        ax.set_xlabel(r"$\eta$")

    fig.tight_layout()
    _save(fig, "fig_robustness.pdf")


def fig_calibrated(npz_path: Path):
    """Phase diagram in the calibrated noise V (circular variance).
    Three panels: (a) eta_alpha(V) calibration curves, (b) phi(V),
    (c) chi(V)."""
    z = np.load(npz_path)
    alphas = z["alphas"]
    V = z["V_grid"]
    eta_table = z["eta_table"]
    phi = z["phi_mean"]
    chi = z["chi"]

    palette = ["#3aa040", "#d76f3a", "#1f4ea1"]
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE_COL[0], 2.6))

    ax = axes[0]
    for ia, a in enumerate(alphas):
        ax.plot(V, eta_table[ia], "o-", color=palette[ia], ms=3, lw=1.0,
                label=fr"$\alpha={a:.1f}$")
    ax.set_xlabel(r"calibrated noise $V$")
    ax.set_ylabel(r"$\eta_\alpha(V)$")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title("noise calibration", fontsize=8)
    axes[0].text(-0.18, 1.04, "(a)", transform=axes[0].transAxes,
                 fontsize=10, fontweight="bold")

    ax = axes[1]
    for ia, a in enumerate(alphas):
        ax.plot(V, phi[ia], "o-", color=palette[ia], ms=3, lw=1.0,
                label=fr"$\alpha={a:.1f}$")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$\langle\varphi\rangle$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7)
    ax.set_title("polar order", fontsize=8)
    axes[1].text(-0.18, 1.04, "(b)", transform=axes[1].transAxes,
                 fontsize=10, fontweight="bold")

    ax = axes[2]
    for ia, a in enumerate(alphas):
        ax.plot(V, chi[ia], "s-", color=palette[ia], ms=3, lw=1.0,
                label=fr"$\alpha={a:.1f}$")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$\chi$")
    ax.legend(fontsize=7)
    ax.set_title("susceptibility", fontsize=8)
    axes[2].text(-0.18, 1.04, "(c)", transform=axes[2].transAxes,
                 fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_calibrated.pdf")


def fig_diffusion(npz_path: Path):
    """Angular and spatial mean-square displacements vs t, log-log,
    one panel each. Three alpha at one near-critical eta = 0.15."""
    z = np.load(npz_path)
    alphas = z["alphas"]
    etas = z["etas"]
    t = z["t"].astype(float)
    msd_theta = z["msd_theta"]
    msd_x = z["msd_x"]

    palette = ["#3aa040", "#d76f3a", "#1f4ea1"]
    ie = int(np.argmin(np.abs(etas - 0.15)))

    fig, axes = plt.subplots(1, 2, figsize=(style.DOUBLE_COL[0], 2.6))

    # Angular MSD
    ax = axes[0]
    for ia, alpha_val in enumerate(alphas):
        ax.loglog(t, msd_theta[ia, ie], "o-",
                  color=palette[ia], ms=3, lw=1.0,
                  label=fr"$\alpha={alpha_val:.1f}$")
    # Reference linear (normal-diffusion) slope
    t_ref = np.geomspace(t[1], t[-1], 50)
    A = msd_theta[-1, ie, len(t)//2] / t[len(t)//2]
    ax.loglog(t_ref, A * t_ref, "k--", lw=0.7, alpha=0.6,
              label=r"slope $1$")
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"$\langle\Delta\theta^2\rangle$")
    ax.legend(fontsize=7)
    ax.set_title(r"angular MSD ($\eta = 0.15$)", fontsize=8)
    axes[0].text(-0.18, 1.04, "(a)", transform=axes[0].transAxes,
                 fontsize=10, fontweight="bold")

    # Spatial MSD
    ax = axes[1]
    for ia, alpha_val in enumerate(alphas):
        ax.loglog(t, msd_x[ia, ie], "o-",
                  color=palette[ia], ms=3, lw=1.0,
                  label=fr"$\alpha={alpha_val:.1f}$")
    # Ballistic reference (slope 2)
    B = msd_x[-1, ie, 4] / t[4]**2
    ax.loglog(t_ref, B * t_ref**2, "k--", lw=0.7, alpha=0.6,
              label=r"slope $2$")
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"$\langle|\Delta\vec{r}|^2\rangle$")
    ax.legend(fontsize=7)
    ax.set_title(r"spatial MSD ($\eta = 0.15$)", fontsize=8)
    axes[1].text(-0.18, 1.04, "(b)", transform=axes[1].transAxes,
                 fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_diffusion.pdf")


def fig_correlations(npz_path: Path):
    """Spatial correlations: C_v(r), correlation length xi(eta), g(r).
    Three panels in one row.
    """
    z = np.load(npz_path)
    alphas = z["alphas"]
    etas = z["etas"]
    centers = z["centers"]
    Cv = z["Cv"]
    g = z["g"]
    xi = z["xi"]

    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE_COL[0], 2.7))
    palette = ["#3aa040", "#d76f3a", "#1f4ea1"]

    # Panel (a): C_v(r) at one near-critical eta per alpha.
    near_crit = {1.0: 0.05, 1.5: 0.10, 2.0: 0.15}
    ax = axes[0]
    for ia, alpha_val in enumerate(alphas):
        eta_t = near_crit[float(alpha_val)]
        ie = int(np.argmin(np.abs(etas - eta_t)))
        ax.semilogy(centers, np.maximum(Cv[ia, ie], 1e-3), "-",
                    color=palette[ia], lw=1.2,
                    label=fr"$\alpha={alpha_val:.1f}, "
                          fr"\eta={etas[ie]:.2f}$")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$C_v(r)$")
    ax.set_xlim(0, 5)
    ax.set_ylim(1e-2, 1.2)
    ax.legend(fontsize=6)
    ax.set_title("velocity correlation", fontsize=8)
    axes[0].text(-0.22, 1.04, "(a)", transform=axes[0].transAxes,
                 fontsize=10, fontweight="bold")

    # Panel (b): correlation length xi(eta), one curve per alpha.
    # Skip eta = 0 (trivial fully-ordered case where C_v is constant).
    ax = axes[1]
    eta_pos = etas > 0
    for ia, alpha_val in enumerate(alphas):
        ax.plot(etas[eta_pos], xi[ia][eta_pos], "o-",
                color=palette[ia], ms=4, lw=1.2,
                label=fr"$\alpha={alpha_val:.1f}$")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"correlation length $\xi$")
    ax.legend(fontsize=7)
    ax.set_title("correlation length vs noise", fontsize=8)
    axes[1].text(-0.22, 1.04, "(b)", transform=axes[1].transAxes,
                 fontsize=10, fontweight="bold")

    # Panel (c): g(r) at one ordered point per alpha.
    ax = axes[2]
    for ia, alpha_val in enumerate(alphas):
        eta_t = {1.0: 0.02, 1.5: 0.05, 2.0: 0.05}[float(alpha_val)]
        ie = int(np.argmin(np.abs(etas - eta_t)))
        ax.plot(centers, g[ia, ie], "-", color=palette[ia], lw=1.2,
                label=fr"$\alpha={alpha_val:.1f}$")
    ax.axvline(0.5, ls=":", c="grey", lw=0.7)
    ax.axvline(0.7, ls=":", c="grey", lw=0.7)
    ax.text(0.52, 0.05, r"$R_r$", fontsize=7, color="grey")
    ax.text(0.72, 0.05, r"$R_a$", fontsize=7, color="grey")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$g(r)$")
    ax.set_xlim(0, 3.5)
    ax.legend(fontsize=7)
    ax.set_title("pair correlation", fontsize=8)
    axes[2].text(-0.22, 1.04, "(c)", transform=axes[2].transAxes,
                 fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_correlations.pdf")


def fig_bands(npz_path: Path):
    """Travelling-band visualisation: top row shows a snapshot per
    alpha (rotated so the polar direction is along +x); bottom row
    shows the time-averaged density profile along that direction.
    """
    z = np.load(npz_path)
    alphas = z["alphas"]
    etas = z["etas"]
    profiles = z["profiles"]
    centers = z["centers"]
    band_idx = z["band_idx"]
    snap_x = z["snap_x"]
    snap_y = z["snap_y"]
    snap_theta = z["snap_theta"]
    L = float(z["params"][1])

    fig, axes = plt.subplots(2, len(alphas),
                             figsize=(style.DOUBLE_COL[0], 4.6),
                             gridspec_kw={"height_ratios": [1.7, 1.0]})

    arrow_len = 0.55
    for ic in range(len(alphas)):
        # --- snapshot rotated to align global flow with +x ---
        x = snap_x[ic]
        y = snap_y[ic]
        th = snap_theta[ic]
        theta_avg = float(np.arctan2(np.sin(th).sum(),
                                     np.cos(th).sum()))
        c, s = np.cos(-theta_avg), np.sin(-theta_avg)
        xr = (c * x - s * y) % L
        yr = (s * x + c * y) % L
        thr = (th - theta_avg + np.pi) % (2 * np.pi) - np.pi

        ax = axes[0, ic]
        u = np.cos(thr)
        v = np.sin(thr)
        ax.quiver(xr, yr, u, v,
                  color=style.PARTICLE_BLUE,
                  scale=1.0 / arrow_len, scale_units="xy",
                  angles="xy", width=0.003,
                  headwidth=2.8, headlength=3.4)
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            fr"$\alpha={alphas[ic]:.1f}$, $\eta={etas[ic]:.2f}$"
            "\n"
            fr"band index $= {band_idx[ic]:.2f}$",
            fontsize=8,
        )

        # --- time-averaged density profile along the flow axis ---
        ax = axes[1, ic]
        norm = profiles[ic] / profiles[ic].mean()
        ax.plot(centers, norm, "-", color=style.PARTICLE_BLUE, lw=1.2)
        ax.axhline(1.0, ls=":", c="grey", lw=0.6)
        ax.set_xlabel(r"$x_\parallel$")
        if ic == 0:
            ax.set_ylabel(r"$\sigma(x_\parallel)/\langle\sigma\rangle$")
        ax.set_xlim(0, L)
        ax.set_ylim(0, max(1.4, 1.1 * norm.max()))
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    _save(fig, "fig_bands.pdf")


def fig_phase_curve(npz_path: Path):
    """Phase boundary eta_c(alpha) extracted from a smooth alpha-grid
    sweep, plus the underlying phi(eta) and chi(eta) curves."""
    z = np.load(npz_path)
    alphas = z["alphas"]
    etas = z["etas"]
    phi = z["phi_mean"]
    chi = z["chi"]
    eta_c_chi = z["eta_c_chi"]
    eta_c_phi = z["eta_c_phi"]

    cmap = plt.cm.viridis
    n_a = len(alphas)
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE_COL[0], 2.6))

    ax = axes[0]
    for ia, a in enumerate(alphas):
        ax.plot(etas, phi[ia], "o-",
                color=cmap(ia / max(1, n_a - 1)),
                ms=3.5, lw=1.0,
                label=fr"$\alpha={a:.2f}$")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\langle\varphi\rangle$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("polar order", fontsize=8)
    axes[0].text(-0.18, 1.04, "(a)", transform=axes[0].transAxes,
                 fontsize=10, fontweight="bold")

    ax = axes[1]
    for ia, a in enumerate(alphas):
        ax.plot(etas, chi[ia], "s-",
                color=cmap(ia / max(1, n_a - 1)),
                ms=3.5, lw=1.0,
                label=fr"$\alpha={a:.2f}$")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\chi$")
    ax.set_title("susceptibility", fontsize=8)
    axes[1].text(-0.18, 1.04, "(b)", transform=axes[1].transAxes,
                 fontsize=10, fontweight="bold")

    ax = axes[2]
    ax.plot(alphas, eta_c_chi, "o-", color="#d76f3a",
            ms=5, lw=1.2, label=r"from $\chi$ peak")
    ax.plot(alphas, eta_c_phi, "s--", color="#1f4ea1",
            ms=5, lw=1.2, label=r"from $\langle\varphi\rangle = 1/2$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\eta_c$")
    ax.legend(fontsize=7)
    ax.set_title("phase boundary", fontsize=8)
    axes[2].text(-0.18, 1.04, "(c)", transform=axes[2].transAxes,
                 fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_phase_curve.pdf")


def fig_adaptive_pilot(npz_path: Path):
    """Pilot comparison of fixed-alpha=1 vs adaptive alpha_i in [1,2].
    Six panels: (top row) phi, chi, U4 vs eta with L overlay for each
    mode; (bottom row) alpha_mean(eta) and alpha_std(eta) for the
    adaptive run, plus the chi_max(L) scaling for both modes."""
    z = np.load(npz_path, allow_pickle=True)
    modes = [str(m) if isinstance(m, str) else m.decode() for m in z["modes"]]
    Ls = z["Ls"]
    etas = z["etas"]
    phi = z["phi"]
    chi = z["chi"]
    binder = z["binder"]
    alpha_mean = z["alpha_mean"]
    alpha_std = z["alpha_std"]

    n_L = len(Ls)
    cmap_fix = plt.cm.Greens(np.linspace(0.4, 0.95, n_L))
    cmap_ada = plt.cm.Oranges(np.linspace(0.4, 0.95, n_L))

    fig, axes = plt.subplots(2, 3, figsize=(style.DOUBLE_COL[0], 4.6))

    panels = [(phi, r"$\langle\varphi\rangle$", "polar order", (-0.02, 1.02)),
              (chi, r"$\chi$", "susceptibility", None),
              (binder, r"$U_4$", "Binder cumulant", None)]

    for k, (arr, ylab, title, ylim) in enumerate(panels):
        ax = axes[0, k]
        for il, L in enumerate(Ls):
            ax.plot(etas, arr[0, il], "o--",
                    color=cmap_fix[il], ms=3, lw=0.9,
                    label=fr"fix, $L={int(L)}$" if k == 0 else None)
            ax.plot(etas, arr[1, il], "s-",
                    color=cmap_ada[il], ms=3, lw=1.0,
                    label=fr"adapt, $L={int(L)}$" if k == 0 else None)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=8)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if title == "Binder cumulant":
            ax.axhline(2.0 / 3.0, ls=":", c="grey", lw=0.6)
        ax.text(-0.18, 1.04, f"({chr(97+k)})",
                transform=ax.transAxes, fontsize=10, fontweight="bold")
    axes[0, 0].legend(fontsize=5, ncol=2, loc="lower left")

    # alpha_mean(eta) at largest L
    ax = axes[1, 0]
    for il, L in enumerate(Ls):
        ax.plot(etas, alpha_mean[il], "o-", color=cmap_ada[il],
                ms=3.5, lw=1.0, label=fr"$L={int(L)}$")
    ax.axhline(1.0, ls=":", c="grey", lw=0.6)
    ax.axhline(2.0, ls=":", c="grey", lw=0.6)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\langle\alpha_i\rangle$")
    ax.set_title("population-mean stability index", fontsize=8)
    ax.legend(fontsize=6)
    ax.text(-0.18, 1.04, "(d)", transform=ax.transAxes,
            fontsize=10, fontweight="bold")

    # alpha_std(eta)
    ax = axes[1, 1]
    for il, L in enumerate(Ls):
        ax.plot(etas, alpha_std[il], "s-", color=cmap_ada[il],
                ms=3.5, lw=1.0, label=fr"$L={int(L)}$")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\sigma(\alpha_i)$")
    ax.set_title("population spread of $\\alpha_i$", fontsize=8)
    ax.legend(fontsize=6)
    ax.text(-0.18, 1.04, "(e)", transform=ax.transAxes,
            fontsize=10, fontweight="bold")

    # chi_max(L) for both modes
    ax = axes[1, 2]
    chi_max_fix = np.array([chi[0, il].max() for il in range(n_L)])
    chi_max_ada = np.array([chi[1, il].max() for il in range(n_L)])
    ax.loglog(Ls, chi_max_fix, "o--", color="#3aa040", ms=4,
              label="fixed $\\alpha=1$")
    ax.loglog(Ls, chi_max_ada, "s-", color="#d76f3a", ms=4,
              label="adaptive")
    if len(Ls) >= 2:
        sf = np.polyfit(np.log(Ls), np.log(chi_max_fix), 1)[0]
        sa = np.polyfit(np.log(Ls), np.log(chi_max_ada), 1)[0]
        ax.text(0.05, 0.95, fr"slope$_{{\rm fix}}={sf:.2f}$",
                transform=ax.transAxes, fontsize=7, color="#3aa040")
        ax.text(0.05, 0.87, fr"slope$_{{\rm ad}}={sa:.2f}$",
                transform=ax.transAxes, fontsize=7, color="#d76f3a")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\chi_{\max}$")
    ax.set_title("FSS of susceptibility", fontsize=8)
    ax.legend(fontsize=6)
    ax.text(-0.18, 1.04, "(f)", transform=ax.transAxes,
            fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_adaptive_pilot.pdf")


def fig_synthesis(data_dir: Path):
    """Single graphical summary: alpha-dependence of six diagnostics
    that quantify the heavy-tail effect on the metric Vicsek-Couzin
    transition. Each panel shows one number per alpha extracted from
    the corresponding sweep.
    """
    fig, axes = plt.subplots(2, 3, figsize=(style.DOUBLE_COL[0], 4.6))
    axes = axes.flatten()
    palette = ["#3aa040", "#d76f3a", "#1f4ea1"]

    def styled(ax, title):
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([1.0, 1.5, 2.0])

    # 1. Hysteresis area (lower = closer to continuous).
    ax = axes[0]
    z = np.load(data_dir / "hysteresis.npz")
    alphas = z["alphas"]
    eta_path = z["eta_path"]
    direction = z["direction"]
    phi_traj = z["phi_traj"]
    # Bin into eta cells, separating up (direction=+1) and down legs.
    eta_grid = np.linspace(eta_path.min(), eta_path.max(), 33)
    centers = 0.5 * (eta_grid[:-1] + eta_grid[1:])
    areas = []
    for ia in range(len(alphas)):
        up_idx = direction > 0
        dn_idx = direction < 0
        up_eta = eta_path[up_idx]
        dn_eta = eta_path[dn_idx]
        up_phi = phi_traj[ia, up_idx]
        dn_phi = phi_traj[ia, dn_idx]
        u_bin = np.array([
            up_phi[(up_eta >= eta_grid[k]) & (up_eta < eta_grid[k + 1])].mean()
            if ((up_eta >= eta_grid[k]) & (up_eta < eta_grid[k + 1])).any()
            else np.nan
            for k in range(len(centers))
        ])
        d_bin = np.array([
            dn_phi[(dn_eta >= eta_grid[k]) & (dn_eta < eta_grid[k + 1])].mean()
            if ((dn_eta >= eta_grid[k]) & (dn_eta < eta_grid[k + 1])).any()
            else np.nan
            for k in range(len(centers))
        ])
        ok = ~(np.isnan(u_bin) | np.isnan(d_bin))
        areas.append(float(np.trapezoid(np.abs(u_bin[ok] - d_bin[ok]),
                                        centers[ok])))
    ax.plot(alphas, areas, "o-", color=palette[0], ms=5, lw=1.2)
    ax.set_ylabel(r"hysteresis area")
    styled(ax, "(a) hysteresis area")

    # 2. Band index from data/bands.npz.
    ax = axes[1]
    zb = np.load(data_dir / "bands.npz")
    ax.plot(zb["alphas"], zb["band_idx"], "s-", color=palette[1],
            ms=5, lw=1.2)
    ax.axhline(1.0, ls=":", c="grey", lw=0.6)
    ax.set_ylabel(r"band index $b$")
    styled(ax, "(b) band amplitude")

    # 3. Correlation length xi at near-critical eta from data/correlations.npz.
    ax = axes[2]
    zc = np.load(data_dir / "correlations.npz")
    alpha_c = zc["alphas"]
    eta_c = zc["etas"]
    xi = zc["xi"]
    near_crit = {1.0: 0.05, 1.5: 0.10, 2.0: 0.15}
    xi_pts = []
    for ia, a in enumerate(alpha_c):
        ie = int(np.argmin(np.abs(eta_c - near_crit[float(a)])))
        xi_pts.append(xi[ia, ie])
    ax.plot(alpha_c, xi_pts, "D-", color=palette[2], ms=5, lw=1.2)
    ax.set_ylabel(r"$\xi$ at $\eta_c(\alpha)$")
    styled(ax, "(c) correlation length")

    # 4. Spatial MSD exponent gamma_x at eta = 0.15 from diffusion.npz.
    ax = axes[3]
    zd = np.load(data_dir / "diffusion.npz")
    a_d = zd["alphas"]
    eta_d = zd["etas"]
    t = zd["t"].astype(float)
    msd_x = zd["msd_x"]
    ie = int(np.argmin(np.abs(eta_d - 0.15)))
    fit = (t >= 50) & (t <= 1500)
    gam_x = []
    for ia in range(len(a_d)):
        s, _ = np.polyfit(np.log(t[fit]), np.log(msd_x[ia, ie][fit]), 1)
        gam_x.append(float(s))
    ax.plot(a_d, gam_x, "v-", color=palette[0], ms=5, lw=1.2)
    ax.axhline(1.0, ls=":", c="grey", lw=0.6,
               label="diffusive")
    ax.axhline(2.0, ls="--", c="grey", lw=0.6,
               label="ballistic")
    ax.set_ylabel(r"$\gamma_x$ at $\eta = 0.15$")
    ax.legend(fontsize=6, loc="lower right")
    styled(ax, "(d) spatial MSD exponent")

    # 5. Effective angular diffusivity D_theta from diffusion.npz.
    ax = axes[4]
    msd_th = zd["msd_theta"]
    D_th = []
    for ia in range(len(a_d)):
        D_th.append(float(msd_th[ia, ie][fit].mean() / t[fit].mean()))
    ax.plot(a_d, D_th, "^-", color=palette[1], ms=5, lw=1.2)
    ax.set_ylabel(r"$D_\theta$ at $\eta = 0.15$")
    styled(ax, "(e) angular diffusivity")

    # 6. Chi_max at L = 45 from fss_sweep.npz.
    ax = axes[5]
    zf = np.load(data_dir / "fss_sweep.npz")
    a_f = zf["alphas"]
    Ls_f = zf["Ls"]
    chi_f = zf["chi"]   # shape (n_L, n_alpha, n_eta)
    il_max = int(np.argmax(Ls_f))
    chi_max = chi_f[il_max].max(axis=1)
    ax.plot(a_f, chi_max, "*-", color=palette[2], ms=7, lw=1.2)
    ax.set_ylabel(fr"$\chi_{{\max}}$ at $L = {int(Ls_f[il_max])}$")
    styled(ax, "(f) FSS susceptibility peak")

    fig.tight_layout()
    _save(fig, "fig_synthesis.pdf")


def fig_orderpdf_k(npz_path_k: Path, npz_path_main: Path):
    """Three-panel comparison of P(<phi>) at alpha = 1 topological,
    L = 30, for k in {4, 6, 10}. The k = 6 panel reuses the trajectory
    from data/orderpdf.npz to avoid recomputing it.
    """
    z_k = np.load(npz_path_k)
    labels_k = z_k["labels"]
    traj_k = z_k["phi_traj"]
    meta_k = z_k["meta"]   # rows = (k, eta)

    z_main = np.load(npz_path_main)
    labels_main = z_main["labels"]
    traj_main = z_main["phi_traj"]
    ic_topo_a1 = int(np.where(labels_main == "topo_a1")[0][0])
    traj_k6 = traj_main[ic_topo_a1]

    panels = []
    for ic, lbl in enumerate(labels_k):
        if lbl == "k4":
            panels.append((4, float(meta_k[ic, 1]), traj_k[ic]))
    panels.append((6, 0.20, traj_k6))
    for ic, lbl in enumerate(labels_k):
        if lbl == "k10":
            panels.append((10, float(meta_k[ic, 1]), traj_k[ic]))

    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE_COL[0], 2.6))
    bins = np.linspace(0.0, 1.0, 60)
    palette = ["#3aa040", "#d76f3a", "#1f4ea1"]

    for k_pan, (k_t, eta, tr) in enumerate(panels):
        ax = axes[k_pan]
        ax.hist(tr, bins=bins, density=True,
                color=palette[k_pan], alpha=0.75,
                edgecolor="black", linewidth=0.4)
        ax.set_title(fr"$k = {int(k_t)}$, $\eta = {eta:.2f}$",
                     fontsize=8)
        ax.set_xlabel(r"$\langle\varphi\rangle$")
        ax.set_ylabel(r"$P(\langle\varphi\rangle)$")
        ax.set_xlim(0, 1)
        ax.text(-0.18, 1.04, f"({chr(97+k_pan)})",
                transform=ax.transAxes,
                fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_orderpdf_k.pdf")


def fig_topo_k(npz_path: Path):
    """Susceptibility curves at k in {4,6,10} for L=15 (dashed) and
    L=30 (solid), at alpha = 1; plus chi_max(k) summary scaling.
    """
    z = np.load(npz_path)
    ks = z["ks"]
    Ls = z["Ls"]
    etas = z["etas"]
    chi = z["chi"]

    palette = ["#3aa040", "#d76f3a", "#1f4ea1"]

    fig, axes = plt.subplots(1, 2, figsize=(style.DOUBLE_COL[0], 2.6))

    ax = axes[0]
    for ik, k_t in enumerate(ks):
        c = palette[ik]
        ax.plot(etas, chi[ik, 0], "o--", color=c, ms=3.5, lw=0.9,
                alpha=0.7, label=fr"$k={int(k_t)}, L=15$")
        ax.plot(etas, chi[ik, 1], "s-", color=c, ms=4, lw=1.1,
                label=fr"$k={int(k_t)}, L=30$")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\chi$")
    ax.set_title(r"$\alpha = 1$: $\chi(\eta)$ for varying $k$",
                 fontsize=8)
    ax.legend(fontsize=6, ncol=2)
    axes[0].text(-0.18, 1.04, "(a)", transform=axes[0].transAxes,
                 fontsize=10, fontweight="bold")

    ax = axes[1]
    chi_max = chi.max(axis=2)  # (k, L)
    slopes = np.log(chi_max[:, 1] / chi_max[:, 0]) \
             / np.log(Ls[1] / Ls[0])
    ax.bar(np.arange(len(ks)), slopes,
           color=palette[: len(ks)], alpha=0.85,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels([f"$k={int(k)}$" for k in ks])
    ax.set_ylabel(r"slope $\log\chi_{\max}(L)/\log L$")
    ax.axhline(1.58, ls=":", c="grey", lw=0.7,
               label=r"$k = 6$ ref ($1.58$)")
    ax.set_title(r"$L = 15 \to 30$ scaling", fontsize=8)
    ax.legend(fontsize=7)
    axes[1].text(-0.18, 1.04, "(b)", transform=axes[1].transAxes,
                 fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_topo_k.pdf")


def fig_orderpdf(npz_path: Path):
    """Probability density of the polar order parameter at the
    FSS-located near-transition point, four cases:
      metric a=2 (first-order ref) and a=1 (continuous ref);
      topological a=2 and a=1 (the new transition).
    Bimodal -> first-order; unimodal -> continuous.
    """
    z = np.load(npz_path)
    labels = z["labels"]
    phi_traj = z["phi_traj"]

    nice_labels = {
        "metric_a2": (r"metric, $\alpha = 2$, $\eta = 0.15$",
                      "first-order ref"),
        "metric_a1": (r"metric, $\alpha = 1$, $\eta = 0.05$",
                      "continuous ref"),
        "topo_a2":   (r"topological, $\alpha = 2$, $\eta = 0.50$",
                      "Ginelli--Chate"),
        "topo_a1":   (r"topological, $\alpha = 1$, $\eta = 0.20$",
                      "new transition"),
    }
    palette = {"metric_a2": "#1f4ea1", "metric_a1": "#3aa040",
               "topo_a2":   "#1f4ea1", "topo_a1":   "#3aa040"}

    fig, axes = plt.subplots(2, 2, figsize=(style.DOUBLE_COL[0], 4.6))
    axes = axes.flatten()
    bins = np.linspace(0.0, 1.0, 60)

    order = ["metric_a2", "metric_a1", "topo_a2", "topo_a1"]
    for k, lbl in enumerate(order):
        ic = int(np.where(labels == lbl)[0][0])
        ax = axes[k]
        ax.hist(phi_traj[ic], bins=bins, density=True,
                color=palette[lbl], alpha=0.7,
                edgecolor="black", linewidth=0.4)
        title, sub = nice_labels[lbl]
        ax.set_title(f"{title}\n{sub}", fontsize=8)
        ax.set_xlabel(r"$\langle\varphi\rangle$")
        ax.set_ylabel(r"$P(\langle\varphi\rangle)$")
        ax.set_xlim(0, 1)
        ax.text(-0.18, 1.04, f"({chr(97+k)})",
                transform=ax.transAxes,
                fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_orderpdf.pdf")


def fig_bands_topo(npz_path: Path):
    """Same layout as fig_bands but for the topological variant."""
    z = np.load(npz_path)
    alphas = z["alphas"]
    etas = z["etas"]
    profiles = z["profiles"]
    centers = z["centers"]
    band_idx = z["band_idx"]
    snap_x = z["snap_x"]
    snap_y = z["snap_y"]
    snap_theta = z["snap_theta"]
    L = float(z["params"][1])

    fig, axes = plt.subplots(2, len(alphas),
                             figsize=(style.DOUBLE_COL[0], 4.6),
                             gridspec_kw={"height_ratios": [1.7, 1.0]})

    arrow_len = 0.55
    for ic in range(len(alphas)):
        x = snap_x[ic]
        y = snap_y[ic]
        th = snap_theta[ic]
        theta_avg = float(np.arctan2(np.sin(th).sum(),
                                     np.cos(th).sum()))
        c, s = np.cos(-theta_avg), np.sin(-theta_avg)
        xr = (c * x - s * y) % L
        yr = (s * x + c * y) % L
        thr = (th - theta_avg + np.pi) % (2 * np.pi) - np.pi

        ax = axes[0, ic]
        u = np.cos(thr)
        v = np.sin(thr)
        ax.quiver(xr, yr, u, v,
                  color=style.PARTICLE_BLUE,
                  scale=1.0 / arrow_len, scale_units="xy",
                  angles="xy", width=0.003,
                  headwidth=2.8, headlength=3.4)
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            fr"$\alpha={alphas[ic]:.1f}$, $\eta={etas[ic]:.2f}$"
            "\n"
            fr"band index $= {band_idx[ic]:.2f}$",
            fontsize=8,
        )

        ax = axes[1, ic]
        norm = profiles[ic] / profiles[ic].mean()
        ax.plot(centers, norm, "-", color=style.PARTICLE_BLUE, lw=1.2)
        ax.axhline(1.0, ls=":", c="grey", lw=0.6)
        ax.set_xlabel(r"$x_\parallel$")
        if ic == 0:
            ax.set_ylabel(r"$\sigma(x_\parallel)/\langle\sigma\rangle$")
        ax.set_xlim(0, L)
        ax.set_ylim(0, max(1.4, 1.1 * norm.max()))
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    _save(fig, "fig_bands_topo.pdf")


def fig_gnf(npz_path: Path):
    """Giant number fluctuations: Var(N_box) vs <N_box> in log-log,
    one curve per (alpha, eta) ordered-phase point, with linear fits
    giving the exponent zeta. Reference line zeta = 1/2 for Poisson.
    The inset shows residuals to the alpha=2 fit, evidencing the
    alpha-independence of the GNF curve.
    """
    z = np.load(npz_path)
    alphas = z["alphas"]
    etas = z["etas"]
    means = z["means"]
    vars_ = z["vars"]

    fig, ax = plt.subplots(figsize=(style.SINGLE_COL[0] * 1.25,
                                    style.SINGLE_COL[1] * 1.15))
    palette = ["#3aa040", "#d76f3a", "#1f4ea1"]
    markers = ["o", "s", "D"]

    # Reference fit: alpha = 2.0 curve (last entry by construction)
    i_ref = int(np.argmax(np.asarray(alphas, dtype=float)))
    m_ref = means[i_ref]; v_ref = vars_[i_ref]
    ok_ref = (m_ref > 0) & (v_ref > 0)
    s_ref, b_ref = np.polyfit(np.log(m_ref[ok_ref]),
                              np.log(v_ref[ok_ref]), 1)

    fits = []
    for i, (alpha_val, eta_val) in enumerate(zip(alphas, etas)):
        m = means[i]; v = vars_[i]
        ok = (m > 0) & (v > 0)
        log_m = np.log(m[ok]); log_v = np.log(v[ok])
        slope, intercept = np.polyfit(log_m, log_v, 1)
        zeta = slope / 2.0
        fits.append((slope, intercept, zeta))
        ax.loglog(m[ok], v[ok], marker=markers[i], ls="",
                  color=palette[i], ms=5, mfc="none", mew=1.2,
                  label=fr"$\alpha={alpha_val:.1f}$, "
                        fr"$\eta={eta_val:.2f}$: $\zeta={zeta:.2f}$")
        m_grid = np.geomspace(m[ok].min(), m[ok].max(), 50)
        ax.loglog(m_grid, np.exp(intercept) * m_grid**slope,
                  "-", color=palette[i], lw=0.8, alpha=0.55)

    all_m = np.concatenate([m[m > 0] for m in means])
    m_grid = np.geomspace(all_m.min(), all_m.max(), 50)
    ax.loglog(m_grid, m_grid, "k:", lw=0.9,
              label=r"Poisson ($\zeta=1/2$)")
    ax.loglog(m_grid, m_grid**1.6 / m_grid[0]**0.6 * m_grid[0],
              "k--", lw=0.7, alpha=0.6,
              label=r"Toner--Tu ($\zeta\simeq 0.8$)")

    ax.set_xlabel(r"$\langle N_\ell\rangle$")
    ax.set_ylabel(r"$\mathrm{Var}(N_\ell)$")
    ax.legend(loc="upper left", fontsize=7)

    # --- Inset: residuals to the alpha=2 fit ---
    ax_in = ax.inset_axes([0.62, 0.10, 0.35, 0.30])
    ax_in.axhline(0.0, color="k", ls="--", lw=0.7, alpha=0.6)
    for i, alpha_val in enumerate(alphas):
        m = means[i]; v = vars_[i]
        ok = (m > 0) & (v > 0)
        v_pred_ref = np.exp(b_ref) * m[ok] ** s_ref
        resid = (v[ok] - v_pred_ref) / v_pred_ref
        ax_in.semilogx(m[ok], resid, marker=markers[i], ls="",
                       color=palette[i], ms=3.5, mfc="none", mew=1.0)
    ax_in.set_ylim(-0.25, 0.25)
    ax_in.set_xlabel(r"$\langle N_\ell\rangle$", fontsize=7,
                     labelpad=1)
    ax_in.set_ylabel(r"$(V - V_{\alpha=2}) / V_{\alpha=2}$",
                     fontsize=7, labelpad=1)
    ax_in.tick_params(labelsize=6, pad=1)
    ax_in.set_title("residuals to $\\alpha=2$ fit", fontsize=7,
                    pad=2)

    fig.tight_layout()
    _save(fig, "fig_gnf.pdf")


def fig_hysteresis(npz_path: Path):
    """Hysteresis loop in (eta, phi) under a slow up/down eta ramp,
    one curve per alpha. The enclosed area is a finite-size-robust
    indicator of a first-order transition.
    """
    z = np.load(npz_path)
    alphas = z["alphas"]
    eta_path = z["eta_path"]
    direction = z["direction"]
    phi_traj = z["phi_traj"]
    T_up = int(z["params"][6])

    # Block-average to smooth single-step fluctuations.
    bw = 80
    def smooth(y):
        n = len(y) // bw
        return np.mean(y[:n * bw].reshape(n, bw), axis=1)

    fig, axes = plt.subplots(1, len(alphas),
                             figsize=(style.DOUBLE_COL[0], 2.6),
                             sharey=True)
    for ia, ax in enumerate(axes):
        eta_up = eta_path[:T_up]
        eta_dn = eta_path[T_up:]
        phi_up = phi_traj[ia, :T_up]
        phi_dn = phi_traj[ia, T_up:]
        ax.plot(smooth(eta_up), smooth(phi_up), "-",
                color="#1f4ea1", lw=1.4, label=r"ramp $\uparrow$")
        ax.plot(smooth(eta_dn), smooth(phi_dn), "-",
                color="#d76f3a", lw=1.4, label=r"ramp $\downarrow$")
        # Hysteresis area as a quantitative tag.
        e_grid = np.linspace(0.0, eta_path.max(), 80)
        phi_up_i = np.interp(e_grid, smooth(eta_up), smooth(phi_up))
        phi_dn_i = np.interp(e_grid, smooth(eta_dn)[::-1],
                             smooth(phi_dn)[::-1])
        loop_area = float(np.trapezoid(phi_up_i - phi_dn_i, e_grid))
        max_gap = float(np.max(phi_up_i - phi_dn_i))
        ax.set_title(fr"$\alpha = {alphas[ia]:.1f}$" "\n"
                     fr"area $= {loop_area:.3f}$, max gap $= {max_gap:.2f}$",
                     fontsize=8)
        ax.set_xlabel(r"noise scale $\eta$")
        if ia == 0:
            ax.set_ylabel(r"polarisation $\langle\varphi\rangle$")
            ax.legend(loc="upper right", fontsize=7)
        ax.set_ylim(-0.02, 1.05)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    _save(fig, "fig_hysteresis.pdf")


def fig_3d_phase(npz_path: Path):
    """3D phase portrait phi(eta, R_r, R_a) at 3 values of alpha.
    Top row: 3D scatter, one panel per alpha.
    Bottom row: marginalised 2D projections at the standard alpha=2.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    z = np.load(npz_path)
    alphas = z["alphas"]
    etas = z["etas"]
    R_rs = z["R_rs"]
    R_as = z["R_as"]
    phi = z["phi"]   # (n_alpha, n_eta, n_Rr, n_Ra), NaN where R_a <= R_r

    n_alpha = len(alphas)
    fig = plt.figure(figsize=(style.DOUBLE_COL[0], 6.2))
    gs = fig.add_gridspec(2, n_alpha, hspace=0.32, wspace=0.32,
                          left=0.07, right=0.88, top=0.95, bottom=0.08)
    cmap_name = "viridis"

    # ---- Top row: 3D scatters ----
    E_g, R_g, A_g = np.meshgrid(etas, R_rs, R_as, indexing="ij")
    for j, alpha_val in enumerate(alphas):
        ax = fig.add_subplot(gs[0, j], projection="3d")
        phi_grid = phi[j]
        valid = ~np.isnan(phi_grid)
        sc = ax.scatter(
            E_g[valid], R_g[valid], A_g[valid],
            c=phi_grid[valid], cmap=cmap_name, vmin=0.0, vmax=1.0,
            s=14, alpha=0.9, edgecolors="none",
        )
        ax.set_xlabel(r"$\eta$", fontsize=8, labelpad=-1)
        ax.set_ylabel(r"$R_r$", fontsize=8, labelpad=-1)
        ax.set_zlabel(r"$R_a$", fontsize=8, labelpad=-1)
        ax.set_title(fr"$\alpha = {alpha_val:.1f}$", fontsize=9)
        ax.tick_params(labelsize=6, pad=-2)
        ax.view_init(elev=22, azim=-58)

    # ---- Bottom row: 2D projections at standard alpha = 2 ----
    ia_std = int(np.argmin(np.abs(alphas - 2.0)))
    phi_std = phi[ia_std]   # (n_eta, n_Rr, n_Ra)

    # Marginal averages (over the missing axis), ignoring NaNs.
    proj_eRr = np.nanmean(phi_std, axis=2)              # (eta, R_r)
    proj_eRa = np.nanmean(phi_std, axis=1)              # (eta, R_a)
    proj_RrRa = np.nanmean(phi_std, axis=0)             # (R_r, R_a)

    def heatmap(ax, x, y, Z, xlabel, ylabel, title):
        dx = np.diff(x).mean()
        dy = np.diff(y).mean()
        x_edges = np.concatenate([x - dx / 2, [x[-1] + dx / 2]])
        y_edges = np.concatenate([y - dy / 2, [y[-1] + dy / 2]])
        im = ax.pcolormesh(x_edges, y_edges, Z.T,
                           cmap=cmap_name, vmin=0.0, vmax=1.0,
                           shading="auto")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
        return im

    ax_b0 = fig.add_subplot(gs[1, 0])
    im = heatmap(ax_b0, etas, R_rs, proj_eRr,
                 r"$\eta$", r"$R_r$",
                 r"avg over $R_a$ ($\alpha=2$)")
    ax_b1 = fig.add_subplot(gs[1, 1])
    heatmap(ax_b1, etas, R_as, proj_eRa,
            r"$\eta$", r"$R_a$",
            r"avg over $R_r$ ($\alpha=2$)")
    ax_b2 = fig.add_subplot(gs[1, 2])
    heatmap(ax_b2, R_rs, R_as, proj_RrRa,
            r"$R_r$", r"$R_a$",
            r"avg over $\eta$ ($\alpha=2$)")
    # Show R_a > R_r constraint as a line.
    rr = np.linspace(R_rs.min(), R_rs.max(), 50)
    ax_b2.plot(rr, rr, "w--", lw=0.7, alpha=0.85)

    cbar_ax = fig.add_axes([0.91, 0.10, 0.018, 0.32])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"polar order $\langle\varphi\rangle$", fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    # Subfigure tags.
    fig.text(0.02, 0.95, "(a)", fontsize=10, fontweight="bold")
    fig.text(0.02, 0.46, "(b)", fontsize=10, fontweight="bold")

    _save(fig, "fig_3d_phase.pdf")


def fig_snapshots():
    cases = [
        ("Ordered", 2.0, 0.05),
        ("Near critical", 2.0, 0.20),
        ("Disordered (heavy tail)", 1.0, 0.10),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE_COL[0], 2.6))
    arrow_len = 0.45
    for ax, (label, alpha, eta) in zip(axes, cases):
        sim, phi_samples = _run_for_snapshot(
            alpha=alpha, eta=eta, N=500, L=15.0,
            n_warm=600, n_meas=200, seed=7,
        )
        phi_mean = float(phi_samples.mean())
        phi_std = float(phi_samples.std())
        u = np.cos(sim.theta)
        v = np.sin(sim.theta)
        ax.quiver(
            sim.x, sim.y, u, v,
            color=style.PARTICLE_BLUE,
            scale=1.0 / arrow_len, scale_units="xy",
            angles="xy", width=0.004, headwidth=3.5, headlength=4.0,
        )
        ax.set_xlim(0, sim.p.L)
        ax.set_ylim(0, sim.p.L)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{label}\n"
            fr"$\alpha={alpha:.1f}$, $\eta={eta:.2f}$, "
            fr"$\langle\varphi\rangle={phi_mean:.2f}\pm{phi_std:.2f}$",
            fontsize=8,
        )

    fig.tight_layout()
    _save(fig, "fig_snapshots.pdf")


def fig_model_schematic():
    """Schematic of the zonal Vicsek-Couzin update rule."""
    from matplotlib.patches import Wedge, Patch

    # Zones enlarged by 20%; neighbours and arrows scaled to keep
    # categorisation correct.
    R_r = 0.5 * 1.2
    R_a = 0.7 * 1.2
    half_blind = 15.0  # degrees (full blind sector = 30 deg)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))

    rep_color = "#e07b7b"
    ali_color = "#9bb8de"
    blind_color = "#bfbfbf"

    # Visible cone goes from -165 deg to +165 deg (CCW), leaving a 30 deg
    # rear blind sector centred on 180 deg.
    vis_t1 = -180 + half_blind
    vis_t2 = 180 - half_blind

    rep = Wedge((0, 0), R_r, vis_t1, vis_t2,
                facecolor=rep_color, alpha=0.55,
                edgecolor="#9c3a3a", lw=0.9, zorder=1)
    ali = Wedge((0, 0), R_a, vis_t1, vis_t2, width=R_a - R_r,
                facecolor=ali_color, alpha=0.55,
                edgecolor="#3a4a78", lw=0.9, zorder=1)
    blind = Wedge((0, 0), R_a, 180 - half_blind, 180 + half_blind,
                  facecolor=blind_color, alpha=0.65, hatch="///",
                  edgecolor="#666", lw=0.7, zorder=2)
    for patch in (rep, ali, blind):
        ax.add_patch(patch)

    # Focal particle i: arrow at origin pointing +x.
    head_len = 0.20 * 1.2
    ax.annotate("", xy=(head_len, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=style.PARTICLE_BLUE,
                                lw=2.6), zorder=6)
    ax.scatter([0], [0], s=90, color=style.PARTICLE_BLUE,
               edgecolor="white", lw=0.8, zorder=7)
    ax.text(0.05, -0.13, r"$i$", fontsize=13, fontweight="bold", zorder=7)
    ax.text(head_len + 0.02, 0.05, r"$\vec e_i(t)$", fontsize=12,
            color=style.PARTICLE_BLUE, zorder=7)

    arrow_len = 0.14 * 1.2

    def neighbour(x, y, theta_deg, label, color=style.PARTICLE_BLUE,
                  alpha=1.0):
        th = np.deg2rad(theta_deg)
        ax.annotate("",
                    xy=(x + arrow_len * np.cos(th), y + arrow_len * np.sin(th)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                    alpha=alpha), zorder=6)
        ax.scatter([x], [y], s=42, color=color, alpha=alpha,
                   edgecolor="white", lw=0.6, zorder=7)
        ax.text(x + 0.05, y + 0.08, label, fontsize=11, alpha=alpha,
                zorder=7)

    # Repulsion neighbour: triggers a turn-away vector.
    j1 = (-0.18 * 1.2, 0.22 * 1.2)
    neighbour(*j1, theta_deg=90, label=r"$j_1$")
    nrm = np.hypot(*j1)
    away = (-j1[0] / nrm * 0.30 * 1.2, -j1[1] / nrm * 0.30 * 1.2)
    ax.annotate("", xy=away, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#9c3a3a", lw=1.5,
                                ls=(0, (3, 2))), zorder=6)
    ax.text(away[0] + 0.02, away[1] - 0.06, "repulse", fontsize=10,
            color="#9c3a3a", style="italic", zorder=7)

    # Alignment neighbours.
    neighbour(0.46 * 1.2, 0.42 * 1.2, theta_deg=20, label=r"$j_2$")
    neighbour(-0.50 * 1.2, -0.34 * 1.2, theta_deg=200, label=r"$j_3$")

    # Blind-sector neighbour (rear): visible position but ignored.
    neighbour(-0.55 * 1.2, 0.06 * 1.2, theta_deg=0, label=r"$j_b$",
              color="#666", alpha=0.55)

    # Outside-R_a neighbour: position out of perception range.
    neighbour(0.85 * 1.2, 0.55 * 1.2, theta_deg=60, label=r"$j_\infty$",
              color="#666", alpha=0.55)

    # Radius labels.
    ax.annotate(r"$R_r$",
                xy=(R_r * np.cos(np.deg2rad(-50)),
                    R_r * np.sin(np.deg2rad(-50))),
                xytext=(0.66, -1.14), fontsize=12,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#444"))
    ax.annotate(r"$R_a$",
                xy=(R_a * np.cos(np.deg2rad(-30)),
                    R_a * np.sin(np.deg2rad(-30))),
                xytext=(1.26, -0.74), fontsize=12,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#444"))

    # Blind angle annotation.
    ax.annotate(r"blind sector $\beta$",
                xy=(-0.74, 0.19), xytext=(-1.74, 0.74), fontsize=12,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#444"))

    # Legend (zone colours).
    handles = [
        Patch(facecolor=rep_color, alpha=0.55, edgecolor="#9c3a3a",
              label=r"Repulsion ($d<R_r$)"),
        Patch(facecolor=ali_color, alpha=0.55, edgecolor="#3a4a78",
              label=r"Alignment ($R_r \leq d < R_a$)"),
        Patch(facecolor=blind_color, alpha=0.65, hatch="///",
              edgecolor="#666", label="Blind sector (ignored)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10,
              framealpha=0.92)

    ax.set_xlim(-1.86, 1.86)
    ax.set_ylim(-1.44, 1.44)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Zonal Vicsek--Couzin update with vision cone",
                 fontsize=12)

    fig.tight_layout()
    _save(fig, "fig_model_schematic.pdf")


def main():
    fig_noise_pdf()
    fig_model_schematic()
    fig_snapshots()
    # fig_phase (single-L phi/chi/U4 vs eta) is superseded by fig_fss.

    npz_3d = DATA / "sweep_3d.npz"
    if npz_3d.exists():
        fig_3d_phase(npz_3d)
    else:
        print(f"[warn] {npz_3d} not found -- run run_3d_sweep.py first")

    npz_hys = DATA / "hysteresis.npz"
    if npz_hys.exists():
        fig_hysteresis(npz_hys)
    else:
        print(f"[warn] {npz_hys} not found -- run run_hysteresis.py first")

    npz_gnf = DATA / "gnf.npz"
    if npz_gnf.exists():
        fig_gnf(npz_gnf)
    else:
        print(f"[warn] {npz_gnf} not found -- run run_gnf.py first")

    npz_bands = DATA / "bands.npz"
    if npz_bands.exists():
        fig_bands(npz_bands)
    else:
        print(f"[warn] {npz_bands} not found -- run run_bands.py first")

    npz_corr = DATA / "correlations.npz"
    if npz_corr.exists():
        fig_correlations(npz_corr)
    else:
        print(f"[warn] {npz_corr} not found -- "
              "run run_correlations.py first")

    npz_cl = DATA / "clusters.npz"
    if npz_cl.exists():
        fig_clusters(npz_cl)
    else:
        print(f"[warn] {npz_cl} not found -- run run_clusters.py first")

    npz_diff = DATA / "diffusion.npz"
    if npz_diff.exists():
        fig_diffusion(npz_diff)
    else:
        print(f"[warn] {npz_diff} not found -- run run_diffusion.py first")

    npz_cal = DATA / "calibrated_sweep.npz"
    if npz_cal.exists():
        fig_calibrated(npz_cal)
    else:
        print(f"[warn] {npz_cal} not found -- run run_calibrated.py first")

    npz_rob = DATA / "robustness.npz"
    if npz_rob.exists():
        fig_robustness(npz_rob)
    else:
        print(f"[warn] {npz_rob} not found -- run run_robustness.py first")

    npz_topo = DATA / "topo_fss.npz"
    if npz_topo.exists():
        fig_topological(npz_topo)
    else:
        print(f"[warn] {npz_topo} not found -- "
              "run run_topological.py first")

    npz_bt = DATA / "bands_topo.npz"
    if npz_bt.exists():
        fig_bands_topo(npz_bt)
    else:
        print(f"[warn] {npz_bt} not found -- "
              "run run_bands_topo.py first")

    npz_op = DATA / "orderpdf.npz"
    if npz_op.exists():
        fig_orderpdf(npz_op)
    else:
        print(f"[warn] {npz_op} not found -- "
              "run run_orderpdf.py first")

    npz_tk = DATA / "topo_k_scan.npz"
    if npz_tk.exists():
        fig_topo_k(npz_tk)
    else:
        print(f"[warn] {npz_tk} not found -- run run_topo_k.py first")

    npz_pc = DATA / "phase_curve.npz"
    if npz_pc.exists():
        fig_phase_curve(npz_pc)
    else:
        print(f"[warn] {npz_pc} not found -- run run_phasecurve.py first")

    npz_opk = DATA / "orderpdf_k.npz"
    npz_op = DATA / "orderpdf.npz"
    if npz_opk.exists() and npz_op.exists():
        fig_orderpdf_k(npz_opk, npz_op)
    elif not npz_opk.exists():
        print(f"[warn] {npz_opk} not found -- run run_orderpdf_k.py")

    npz_ap = DATA / "adaptive_pilot.npz"
    if npz_ap.exists():
        fig_adaptive_pilot(npz_ap)
    else:
        print(f"[warn] {npz_ap} not found -- run run_adaptive_pilot.py")

    # Synthesis figure: needs hysteresis, bands, correlations,
    # diffusion, and fss_sweep all present.
    needed = ["hysteresis.npz", "bands.npz", "correlations.npz",
              "diffusion.npz", "fss_sweep.npz"]
    if all((DATA / n).exists() for n in needed):
        fig_synthesis(DATA)
    else:
        missing = [n for n in needed if not (DATA / n).exists()]
        print(f"[warn] synthesis missing: {missing}")

    npz_fss = DATA / "fss_sweep.npz"
    if npz_fss.exists():
        fig_fss(npz_fss)
    else:
        print(f"[warn] {npz_fss} not found -- run run_fss_sweep.py first")


if __name__ == "__main__":
    main()
