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
V1_ROOT = HERE.parent.parent if HERE.parent.name == "notes" else HERE.parent
DATA = V1_ROOT / "data"
FIG = V1_ROOT / "figures"
FIG.mkdir(exist_ok=True)


style.apply()


def _save(fig, name: str):
    out = FIG / name
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"saved: {out}")


def fig_noise_pdf():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
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
    fig.suptitle(r"(b) Symmetric $\alpha$-stable noise pdfs",
                 fontsize=13.5, y=0.97)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
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


def fig_topological(npz_path: Path):
    """phi(eta), chi(eta), U4(eta) for the topological variant on
    the harmonised 7-size lever {15, 22, 30, 45, 64, 91, 128} and
    3-alpha grid {1, 1.5, 2}. Base data has alphas in {1, 2} and
    Ls in {15..45}; topo_fss_a15.npz supplies the alpha = 1.5 slice
    over the small-L lever, and topo_L{64,91,128}_a3.npz extend
    each in L (with topo_L64.npz fallback if the *_a3 variant is
    absent)."""
    z = np.load(npz_path)
    alphas = list(z["alphas"])
    Ls = list(z["Ls"])
    etas = z["etas"]
    phi = z["phi"]
    chi = z["chi"]
    binder = z["binder"]
    if phi.ndim == 3:
        phi = phi[..., None]; chi = chi[..., None]; binder = binder[..., None]
    n_seeds = phi.shape[-1]

    # --- alpha = 1.5 slice for the small-L lever ---------------------
    a15_path = npz_path.parent / "topo_fss_a15.npz"
    if a15_path.exists():
        zA = np.load(a15_path)
        if list(zA["Ls"]) == Ls:
            # (n_L, n_eta, n_seed) -> add as a new alpha axis (insert
            # so the final alpha order is sorted ascending: 1, 1.5, 2).
            n_use = min(n_seeds, zA["phi"].shape[-1])
            phi = phi[..., :n_use]; chi = chi[..., :n_use]
            binder = binder[..., :n_use]
            new_phi = zA["phi"][None, :, :, :n_use]
            new_chi = zA["chi"][None, :, :, :n_use]
            new_U4 = zA["U4"][None, :, :, :n_use]
            ins = next(i for i, a in enumerate(alphas) if a > 1.5)
            alphas = alphas[:ins] + [1.5] + alphas[ins:]
            phi = np.concatenate([phi[:ins], new_phi, phi[ins:]], axis=0)
            chi = np.concatenate([chi[:ins], new_chi, chi[ins:]], axis=0)
            binder = np.concatenate(
                [binder[:ins], new_U4, binder[ins:]], axis=0)
            n_seeds = n_use

    # --- L extensions: prefer the 3-alpha files, fall back to 2-alpha
    for L_ext in (64, 91, 128):
        p3 = npz_path.parent / f"topo_L{L_ext}_a3.npz"
        p2 = npz_path.parent / f"topo_L{L_ext}.npz"
        p = p3 if p3.exists() else (p2 if p2.exists() else None)
        if p is None:
            continue
        zE = np.load(p)
        ext_alphas = list(zE["alphas"])
        n_use = min(n_seeds, zE["phi"].shape[-1])
        phi = phi[..., :n_use]; chi = chi[..., :n_use]
        binder = binder[..., :n_use]
        # zE arrays: (alpha, eta, seed); need (alpha, 1, eta, seed)
        # then concat along L axis. Map to the global alpha order.
        e_phi = np.full((len(alphas), 1, len(etas), n_use), np.nan)
        e_chi = np.full_like(e_phi, np.nan)
        e_U4 = np.full_like(e_phi, np.nan)
        u4_key = "U4" if "U4" in zE.files else "binder"
        for ia_e, a_e in enumerate(ext_alphas):
            if a_e in alphas:
                ia = alphas.index(a_e)
                e_phi[ia, 0] = zE["phi"][ia_e, :, :n_use]
                e_chi[ia, 0] = zE["chi"][ia_e, :, :n_use]
                e_U4[ia, 0] = zE[u4_key][ia_e, :, :n_use]
        Ls.append(float(zE["L"]))
        phi = np.concatenate([phi, e_phi], axis=1)
        chi = np.concatenate([chi, e_chi], axis=1)
        binder = np.concatenate([binder, e_U4], axis=1)
        n_seeds = n_use

    alphas = np.array(alphas)
    Ls = np.array(Ls)

    palette = {1.0: "#3aa040", 1.5: "#c2643a", 2.0: "#1f4ea1"}
    n_L = len(Ls)
    cmap_a1 = plt.cm.Greens(np.linspace(0.35, 0.95, n_L))
    cmap_a15 = plt.cm.Oranges(np.linspace(0.35, 0.95, n_L))
    cmap_a2 = plt.cm.Blues(np.linspace(0.35, 0.95, n_L))
    L_colors = {1.0: cmap_a1, 1.5: cmap_a15, 2.0: cmap_a2}
    markers = ["o", "s", "D", "^", "v", "P", "X"]

    fig = plt.figure(figsize=(style.DOUBLE_COL[0] * 1.25, 5.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.0, 2.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_leg = fig.add_subplot(gs[:, 2])
    ax_leg.axis("off")
    axes = [ax_a, ax_b, ax_c, ax_d]

    panels = [
        (ax_a, phi, r"$\langle\varphi\rangle$", "polar order"),
        (ax_b, chi, r"$\chi$", "susceptibility"),
        (ax_c, binder, r"$U_4$", "Binder cumulant"),
    ]
    for ax, arr, ylabel, title in panels:
        for ia, a in enumerate(alphas):
            for il, L in enumerate(Ls):
                m = arr[ia, il].mean(axis=-1)
                se = arr[ia, il].std(axis=-1) / max(1.0, np.sqrt(n_seeds))
                col = L_colors[float(a)][il]
                ax.plot(etas, m,
                        marker=markers[il % len(markers)], ls="-",
                        color=col, ms=4.5, lw=1.1,
                        label=fr"$\alpha={a:g},\,L={int(L)}$")
                ax.plot(etas, m - se, ls="--", color=col, lw=0.6,
                        alpha=0.7)
                ax.plot(etas, m + se, ls="--", color=col, lw=0.6,
                        alpha=0.7)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if title == "polar order":
            ax.set_ylim(-0.02, 1.02)
        if title == "Binder cumulant":
            ax.axhline(2.0 / 3.0, ls=":", c="grey", lw=0.7)
    # Shared (alpha, L) key, transferred out of panel (a) into the
    # full-height right-hand panel.
    handles, labels = ax_a.get_legend_handles_labels()
    leg = ax_leg.legend(handles, labels, loc="center", fontsize=8,
                        frameon=True, edgecolor="0.35", fancybox=False,
                        ncol=1, handlelength=1.6, labelspacing=0.55,
                        borderaxespad=0.0, borderpad=0.9,
                        title=r"legend, panels (a)--(d)")
    leg.get_title().set_fontsize(9)
    leg.get_title().set_fontweight("bold")

    # Panel (d): chi_max(L) with bootstrap-CI slope.
    ax = ax_d
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
                    label=fr"$\alpha={a:g}$: $L^{{{s_mean:.2f}\pm{s_se:.2f}}}$")
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
    ax.set_title("FSS of susceptibility")
    ax.legend(loc="upper left", fontsize=7)

    for k, ax in enumerate(axes):
        ax.text(-0.18, 1.04, f"({chr(97+k)})", transform=ax.transAxes,
                fontsize=11, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_topological.pdf")


def fig_robustness(npz_path: Path):
    """Robustness of the phi(eta) curve under v_0 and sigma at
    alpha = 1 and 2. Each row is one axis, left column alpha=2, right
    column alpha=1. Mean across seeds with +/- SEM ribbons.
    (The blind-angle row has been removed with the blind sector.)
    """
    z = np.load(npz_path)
    etas = z["etas"]
    alphas = z["alphas"]
    grids = {
        "v0": (z["v0_grid"], z["phi_v0"], r"$v_0$"),
        "sigma": (z["sigma_grid"], z["phi_sigma"], r"$\sigma$"),
    }
    # Detect per-seed shape (..., n_seeds); fall back to mean-only.
    has_seed_axis = next(iter(grids.values()))[1].ndim == 4

    fig, axes = plt.subplots(2, 2, figsize=(style.DOUBLE_COL[0], 4.6),
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
                    ax.plot(etas, m - se, ls="--", color=c, lw=0.6,
                            alpha=0.7)
                    ax.plot(etas, m + se, ls="--", color=c, lw=0.6,
                            alpha=0.7)
                else:
                    ax.plot(etas, phi[ja, ig], "o-",
                            color=c, ms=3, lw=1.0, label=f"{val:g}")
            if irow == 0:
                ax.set_title(fr"$\alpha = {alpha_val:.1f}$", fontsize=9)
            ax.set_ylim(-0.02, 1.02)
            ax.legend(title=label, fontsize=7, title_fontsize=7,
                      loc="upper right")
        axes[irow, 0].set_ylabel(r"$\langle\varphi\rangle$")

    for ax in axes[-1]:
        ax.set_xlabel(r"$\eta$")

    fig.tight_layout()
    _save(fig, "fig_robustness.pdf")


def fig_diffusion(npz_path: Path, clusters_path: Path):
    """1x4 single-particle and cluster summary at one near-critical
    operating point. (a) angular MSD and (b) spatial MSD vs t at
    eta = 0.15, each curve annotated with an asymptotic-window slope
    and point-bootstrap CI; (c) cluster-size distribution P(s) at one
    near-critical eta per alpha; (d) largest-cluster fraction
    <s_max>/N vs eta.
    """
    z = np.load(npz_path)
    alphas = z["alphas"]
    etas = z["etas"]
    t = z["t"].astype(float)
    msd_theta = z["msd_theta"]
    msd_x = z["msd_x"]

    zc = np.load(clusters_path)
    c_alphas = zc["alphas"]
    c_etas = zc["etas"]
    hist_bins = zc["hist_bins"]
    hist = zc["hist"]
    smax_frac = zc["smax_frac"]
    centers = np.sqrt(hist_bins[:-1] * hist_bins[1:])
    widths = np.diff(hist_bins)

    palette = ["#3aa040", "#d76f3a", "#1f4ea1"]
    ie = int(np.argmin(np.abs(etas - 0.15)))

    rng = np.random.default_rng(0)
    n_boot = 800

    def _slope_with_ci(t_arr, y_arr, t_lo, t_hi):
        idx = np.where((t_arr >= t_lo) & (t_arr <= t_hi))[0]
        s, _ = np.polyfit(np.log(t_arr[idx]), np.log(y_arr[idx]), 1)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            sample = rng.choice(idx, size=len(idx), replace=True)
            boot[b] = np.polyfit(np.log(t_arr[sample]),
                                 np.log(y_arr[sample]), 1)[0]
        return float(s), float(boot.std())

    fig, axes = plt.subplots(1, 4,
                             figsize=(style.DOUBLE_COL[0] * 1.5, 2.5))

    # Panel (a): angular MSD.
    ax = axes[0]
    t_lo, t_hi = 50.0, t[-1]
    for ia, alpha_val in enumerate(alphas):
        s, se = _slope_with_ci(t, msd_theta[ia, ie], t_lo, t_hi)
        ax.loglog(t, msd_theta[ia, ie], "o-",
                  color=palette[ia], ms=1.5, lw=1.0,
                  label=fr"$\alpha={alpha_val:.1f}$: "
                        fr"slope$\,{s:.2f}\!\pm\!{se:.2f}$")
    t_ref = np.geomspace(t[1], t[-1], 50)
    A = msd_theta[-1, ie, len(t)//2] / t[len(t)//2]
    ax.loglog(t_ref, A * t_ref, "k:", lw=0.7, alpha=0.5,
              label=r"slope $1$ (ref)")
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"$\langle\Delta\theta^2\rangle$")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(r"angular MSD ($\eta = 0.15$)", fontsize=9)
    ax.text(-0.24, 1.04, "(a)", transform=ax.transAxes,
            fontsize=10, fontweight="bold")

    # Panel (b): spatial MSD.
    ax = axes[1]
    for ia, alpha_val in enumerate(alphas):
        s, se = _slope_with_ci(t, msd_x[ia, ie], t_lo, t_hi)
        ax.loglog(t, msd_x[ia, ie], "o-",
                  color=palette[ia], ms=1.5, lw=1.0,
                  label=(fr"$\alpha={alpha_val:.1f}$: "
                         r"$\gamma_x\!=\!" + fr"{s:.2f}\!\pm\!{se:.2f}$"))
    B = msd_x[-1, ie, 4] / t[4]**2
    ax.loglog(t_ref, B * t_ref**2, "k:", lw=0.7, alpha=0.5,
              label=r"slope $2$ (ref)")
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"$\langle|\Delta\vec{r}|^2\rangle$")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(r"spatial MSD ($\eta = 0.15$)", fontsize=9)
    ax.text(-0.24, 1.04, "(b)", transform=ax.transAxes,
            fontsize=10, fontweight="bold")

    near_crit = {1.0: 0.05, 1.5: 0.10, 2.0: 0.15}

    # Panel (c): cluster-size distribution P(s).
    ax = axes[2]
    for ia, alpha_val in enumerate(c_alphas):
        eta_t = near_crit[float(alpha_val)]
        je = int(np.argmin(np.abs(c_etas - eta_t)))
        h = hist[ia, je]
        Ps = h / (h.sum() * widths)
        ok = (h > 0) & (centers <= 200)
        ax.loglog(centers[ok], Ps[ok], "o-",
                  color=palette[ia], ms=2, lw=1.0,
                  label=fr"$\alpha={alpha_val:.1f}, "
                        fr"\eta={c_etas[je]:.2f}$")
    ref = np.geomspace(2.0, 80.0, 50)
    ax.loglog(ref, ref**-1.7 * 0.5, "k--", lw=0.7, alpha=0.7,
              label=r"slope $-1.7$")
    ax.set_xlabel(r"cluster size $s$")
    ax.set_ylabel(r"$P(s)$")
    ax.legend(fontsize=7)
    ax.set_title("cluster-size distribution", fontsize=9)
    ax.text(-0.24, 1.04, "(c)", transform=ax.transAxes,
            fontsize=10, fontweight="bold")

    # Panel (d): largest-cluster fraction.
    ax = axes[3]
    for ia, alpha_val in enumerate(c_alphas):
        ax.plot(c_etas, smax_frac[ia], "o-", color=palette[ia],
                ms=2, lw=1.2, label=fr"$\alpha={alpha_val:.1f}$")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\langle s_{\max}\rangle / N$")
    ymin, ymax = float(smax_frac.min()), float(smax_frac.max())
    pad = 0.08 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("largest-cluster fraction", fontsize=9)
    ax.text(-0.24, 1.04, "(d)", transform=ax.transAxes,
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
    ax.legend(fontsize=7)
    ax.set_title("velocity correlation", fontsize=9)
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
    ax.set_title("correlation length vs noise", fontsize=9)
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
    ax.set_title("pair correlation", fontsize=9)
    axes[2].text(-0.22, 1.04, "(c)", transform=axes[2].transAxes,
                 fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_correlations.pdf")


def fig_bands(npz_path: Path):
    """Travelling-band visualisation: top row shows a snapshot per
    alpha (rotated so the polar direction is along +x), with the
    same particle glyph as Fig.~\\ref{fig:order_snapshots} (dot at
    the position, thin shaft with a filled triangular head) and a
    circular 5x zoom inset in the top-right corner; bottom row
    shows the time-averaged density profile along the flow
    direction."""
    from matplotlib.patches import Circle, FancyArrowPatch

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

    cmap = plt.get_cmap("twilight")
    # Main-panel glyph drawn 30% larger and thicker than the
    # zoom-inset glyph so it stays legible at the full-box view.
    arr_len = 2.08 * (L / 91.0)
    rng = np.random.default_rng(0)
    n_show = min(int(round(0.18 * L * L)), 1800)

    # Inset (5x zoom) parameters. The source patch is picked
    # adaptively per panel (the densest L/10 disc) so the zoom
    # always lands on a populated region.
    inset_side_frac = 0.50
    pad = 0.02
    real_half = L / 20.0    # source diameter = L/10 (5x zoom)

    # Common tight y-window for the density profiles (panels d-f),
    # hugging the data so no empty margin shows.
    _norms = [profiles[i] / profiles[i].mean()
              for i in range(len(alphas))]
    _half = max(max(n.max() - 1.0, 1.0 - n.min()) for n in _norms)
    prof_lo, prof_hi = 1.0 - _half - 0.015, 1.0 + _half + 0.015

    for ic in range(len(alphas)):
        # --- snapshot rotated to align global flow with +x ---
        x_all = snap_x[ic]
        y_all = snap_y[ic]
        th_all = snap_theta[ic]
        theta_avg = float(np.arctan2(np.sin(th_all).sum(),
                                     np.cos(th_all).sum()))
        c, s = np.cos(-theta_avg), np.sin(-theta_avg)
        xr_all = (c * x_all - s * y_all) % L
        yr_all = (s * x_all + c * y_all) % L
        thr_all = (th_all - theta_avg + np.pi) % (2 * np.pi) - np.pi

        ax = axes[0, ic]

        # Subsample for the main panel.
        idx = rng.choice(xr_all.size,
                         size=min(n_show, xr_all.size),
                         replace=False)
        x = xr_all[idx]; y = yr_all[idx]; th = thr_all[idx]
        col_val = (th + np.pi) / (2 * np.pi)

        ax.scatter(x, y, s=0.78, c=col_val, cmap=cmap,
                   vmin=0.0, vmax=1.0,
                   edgecolors="white", linewidths=0.104, zorder=3)
        for xi, yi, thi, cv in zip(x, y, th, col_val):
            colour = cmap(cv)
            a = FancyArrowPatch(
                (xi, yi),
                (xi + arr_len * np.cos(thi),
                 yi + arr_len * np.sin(thi)),
                arrowstyle="-|>", color=colour,
                lw=0.2925, mutation_scale=2.6,
                shrinkA=0.0, shrinkB=0.0, zorder=2)
            ax.add_patch(a)

        # --- circular 5x zoom inset: densest L/10 disc ---
        best_cnt, cx, cy = -1, 0.5 * L, 0.5 * L
        for fx in np.linspace(0.25, 0.75, 6):
            for fy in np.linspace(0.25, 0.75, 6):
                gx, gy = fx * L, fy * L
                cnt = int(((xr_all - gx) ** 2 + (yr_all - gy) ** 2
                           <= real_half ** 2).sum())
                if cnt > best_cnt:
                    best_cnt, cx, cy = cnt, gx, gy
        in_disc = ((xr_all - cx) ** 2 + (yr_all - cy) ** 2
                   <= real_half ** 2)
        xi_d = xr_all[in_disc]; yi_d = yr_all[in_disc]
        thi_d = thr_all[in_disc]
        cv_d = (thi_d + np.pi) / (2 * np.pi)

        inset_bl = (1 - inset_side_frac - pad,
                    1 - inset_side_frac - pad)
        ax_in = ax.inset_axes([*inset_bl, inset_side_frac,
                               inset_side_frac])
        ax_in.set_xlim(cx - real_half, cx + real_half)
        ax_in.set_ylim(cy - real_half, cy + real_half)
        ax_in.set_aspect("equal")
        ax_in.set_xticks([]); ax_in.set_yticks([])
        ax_in.patch.set_visible(False)
        for sp in ax_in.spines.values():
            sp.set_visible(False)

        bg = Circle((0.5, 0.5), 0.5, transform=ax_in.transAxes,
                    facecolor="white", edgecolor="none", zorder=0)
        ax_in.add_patch(bg)
        clip = Circle((0.5, 0.5), 0.5,
                      transform=ax_in.transAxes,
                      facecolor="none", edgecolor="none")
        ax_in.add_patch(clip)

        if xi_d.size > 0:
            arr_len_in = (2.0 * real_half) * 0.065
            sc = ax_in.scatter(xi_d, yi_d, s=0.6, c=cv_d,
                               cmap=cmap, vmin=0.0, vmax=1.0,
                               edgecolors="white", linewidths=0.08,
                               zorder=4)
            sc.set_clip_path(clip)
            for xx, yy, tt, cc in zip(xi_d, yi_d, thi_d, cv_d):
                colour = cmap(cc)
                a = FancyArrowPatch(
                    (xx, yy),
                    (xx + arr_len_in * np.cos(tt),
                     yy + arr_len_in * np.sin(tt)),
                    arrowstyle="-|>", color=colour,
                    lw=0.225, mutation_scale=2.0,
                    shrinkA=0.0, shrinkB=0.0, zorder=3)
                ax_in.add_patch(a)
                a.set_clip_path(clip)

        outline = Circle((0.5, 0.5), 0.5,
                         transform=ax_in.transAxes,
                         fill=False, edgecolor="black",
                         linewidth=0.9, zorder=10)
        ax_in.add_patch(outline)

        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            fr"({chr(ord('a') + ic)}) "
            fr"$\alpha={alphas[ic]:.1f}$, $\eta={etas[ic]:.2f}$"
            "\n"
            fr"band index $= {band_idx[ic]:.2f}$",
            fontsize=9,
        )

        # --- time-averaged density profile along the flow axis ---
        ax = axes[1, ic]
        norm = profiles[ic] / profiles[ic].mean()
        ax.plot(centers, norm, "-", color=style.PARTICLE_BLUE, lw=1.2)
        ax.fill_between(centers, norm, 1.0, where=(norm > 1.0),
                        color=style.PARTICLE_BLUE, alpha=0.18, lw=0)
        ax.fill_between(centers, norm, 1.0, where=(norm < 1.0),
                        color="#c83a3a", alpha=0.18, lw=0)
        ax.axhline(1.0, ls=":", c="grey", lw=0.6)
        ax.set_xlabel(r"$x_\parallel$")
        if ic == 0:
            ax.set_ylabel(r"$\sigma(x_\parallel)/\langle\sigma\rangle$")
        ax.set_xlim(0, L)
        ax.set_ylim(prof_lo, prof_hi)
        ax.set_title(f"({chr(ord('d') + ic)})", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    _save(fig, "fig_bands.pdf")


def fig_phase_curve(npz_path: Path):
    """Smooth phase boundary (panels a-c from the alpha-grid sweep)
    with the giant-number-fluctuation cross-check as panel (d), in
    a single 1x4 row. Panel (d) reads gnf.npz from the same dir."""
    z = np.load(npz_path)
    alphas = z["alphas"]
    etas = z["etas"]
    phi = z["phi_mean"]
    chi = z["chi"]
    eta_c_chi = z["eta_c_chi"]
    eta_c_phi = z["eta_c_phi"]

    cmap = plt.cm.viridis
    n_a = len(alphas)
    fig, axes = plt.subplots(1, 4,
                             figsize=(style.DOUBLE_COL[0] * 1.3375, 2.5))

    ax = axes[0]
    for ia, a in enumerate(alphas):
        ax.plot(etas, phi[ia], "o-",
                color=cmap(ia / max(1, n_a - 1)),
                ms=1.75, lw=1.0,
                label=fr"$\alpha={a:.2f}$")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\langle\varphi\rangle$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("(a) polar order", fontsize=9)

    ax = axes[1]
    for ia, a in enumerate(alphas):
        ax.plot(etas, chi[ia], "s-",
                color=cmap(ia / max(1, n_a - 1)),
                ms=1.75, lw=1.0)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\chi$")
    ax.set_title("(b) susceptibility", fontsize=9)

    ax = axes[2]
    ax.plot(alphas, eta_c_chi, "o-", color="#d76f3a",
            ms=2.5, lw=1.2, label=r"$\chi$ peak")
    ax.plot(alphas, eta_c_phi, "s--", color="#1f4ea1",
            ms=2.5, lw=1.2, label=r"$\langle\varphi\rangle = 1/2$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\eta_c$")
    ax.legend(fontsize=7)
    ax.set_title("(c) phase boundary", fontsize=9)

    # (d) giant number fluctuations, merged in from gnf.npz
    ax = axes[3]
    zg = np.load(npz_path.parent / "gnf.npz")
    g_alphas = zg["alphas"]
    means = zg["means"]
    vars_ = zg["vars"]
    palette = ["#3aa040", "#d76f3a", "#1f4ea1"]
    markers = ["o", "s", "D"]
    for i, av in enumerate(g_alphas):
        m = means[i]; v = vars_[i]
        ok = (m > 0) & (v > 0)
        slope, intercept = np.polyfit(np.log(m[ok]), np.log(v[ok]), 1)
        zeta = slope / 2.0
        ax.loglog(m[ok], v[ok], marker=markers[i], ls="",
                  color=palette[i], ms=2, mfc="none", mew=1.0,
                  label=fr"$\alpha={av:g}$: $\zeta={zeta:.2f}$")
        mg = np.geomspace(m[ok].min(), m[ok].max(), 40)
        ax.loglog(mg, np.exp(intercept) * mg ** slope, "-",
                  color=palette[i], lw=0.8, alpha=0.55)
    all_m = np.concatenate([m[m > 0] for m in means])
    mg = np.geomspace(all_m.min(), all_m.max(), 40)
    ax.loglog(mg, mg, "k:", lw=0.9, label=r"Poisson")
    ax.loglog(mg, mg ** 1.6 / mg[0] ** 0.6 * mg[0], "k--", lw=0.7,
              alpha=0.6, label=r"Toner--Tu")
    ax.set_xlabel(r"$\langle N_\ell\rangle$")
    ax.set_ylabel(r"$\mathrm{Var}(N_\ell)$")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.set_title("(d) number fluctuations", fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save(fig, "fig_phase_curve.pdf")


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
                     fontsize=9)
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
                 fontsize=9)
    ax.legend(fontsize=7, ncol=2)
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
    ax.axhline(0.96, ls=":", c="grey", lw=0.7,
               label=r"$k = 6$ ref ($0.96$)")
    ax.set_title(r"$L = 15 \to 30$ scaling", fontsize=9)
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

    fig, axes = plt.subplots(1, 4,
                             figsize=(style.DOUBLE_COL[0] * 1.3375, 2.5))
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
        ax.set_title(f"{title}\n{sub}", fontsize=9)
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
            fontsize=9,
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
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    _save(fig, "fig_bands_topo.pdf")


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
        ax.set_xlabel(r"$\eta$", fontsize=9, labelpad=-1)
        ax.set_ylabel(r"$R_r$", fontsize=9, labelpad=-1)
        ax.set_zlabel(r"$R_a$", fontsize=9, labelpad=-1)
        ax.set_title(fr"$\alpha = {alpha_val:.1f}$", fontsize=9)
        ax.tick_params(labelsize=8, pad=-2)
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
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
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
    cbar.set_label(r"polar order $\langle\varphi\rangle$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

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
            fontsize=9,
        )

    fig.tight_layout()
    _save(fig, "fig_snapshots.pdf")


def fig_model_schematic():
    """Schematic of the two-zone Vicsek update rule (no blind sector)."""
    from matplotlib.patches import Wedge, Patch

    # Zones enlarged by 20%; neighbours and arrows scaled to keep
    # categorisation correct. R_r = 0.45 is the canonical operating
    # point of the manuscript (set by the L=22 chi-peak scan).
    R_r = 0.45 * 1.2
    R_a = 0.7 * 1.2

    fig, ax = plt.subplots(figsize=(5.4, 4.4))

    rep_color = "#e07b7b"
    ali_color = "#9bb8de"

    # Full 360 deg vision: both wedges span the entire disk.
    rep = Wedge((0, 0), R_r, 0, 360,
                facecolor=rep_color, alpha=0.55,
                edgecolor="#9c3a3a", lw=0.9, zorder=1)
    ali = Wedge((0, 0), R_a, 0, 360, width=R_a - R_r,
                facecolor=ali_color, alpha=0.55,
                edgecolor="#3a4a78", lw=0.9, zorder=1)
    for patch in (rep, ali):
        ax.add_patch(patch)

    # Focal particle i: arrow at origin pointing +x. All interior
    # elements (circles, arrowheads, labels, vectors) are sized 20%
    # larger than the previous schematic for legibility at 0.33 line
    # width.
    head_len = 0.20 * 1.2 * 1.2
    ax.annotate("", xy=(head_len, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=style.PARTICLE_BLUE,
                                lw=3.1), zorder=6)
    ax.scatter([0], [0], s=108, color=style.PARTICLE_BLUE,
               edgecolor="white", lw=0.96, zorder=7)
    ax.text(0.05, -0.13, r"$i$", fontsize=16, fontweight="bold", zorder=7)
    ax.text(head_len + 0.02, 0.05, r"$\vec e_i(t)$", fontsize=14,
            color=style.PARTICLE_BLUE, zorder=7)

    arrow_len = 0.14 * 1.2 * 1.2

    def neighbour(x, y, theta_deg, label, color=style.PARTICLE_BLUE,
                  alpha=1.0):
        th = np.deg2rad(theta_deg)
        ax.annotate("",
                    xy=(x + arrow_len * np.cos(th), y + arrow_len * np.sin(th)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.92,
                                    alpha=alpha), zorder=6)
        ax.scatter([x], [y], s=50, color=color, alpha=alpha,
                   edgecolor="white", lw=0.72, zorder=7)
        ax.text(x + 0.05, y + 0.08, label, fontsize=13, alpha=alpha,
                zorder=7)

    # Repulsion neighbour: triggers a turn-away vector.
    j1 = (-0.18 * 1.2, 0.22 * 1.2)
    neighbour(*j1, theta_deg=90, label=r"$j_1$")
    nrm = np.hypot(*j1)
    away = (-j1[0] / nrm * 0.30 * 1.2 * 1.2, -j1[1] / nrm * 0.30 * 1.2 * 1.2)
    ax.annotate("", xy=away, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#9c3a3a", lw=1.8,
                                ls=(0, (3, 2))), zorder=6)
    ax.text(away[0] + 0.02, away[1] - 0.06, "repulse", fontsize=12,
            color="#9c3a3a", style="italic", zorder=7)

    # Alignment neighbours (no longer restricted to a forward cone).
    neighbour(0.46 * 1.2, 0.42 * 1.2, theta_deg=20, label=r"$j_2$")
    neighbour(-0.50 * 1.2, -0.34 * 1.2, theta_deg=200, label=r"$j_3$")
    neighbour(-0.55 * 1.2, 0.06 * 1.2, theta_deg=0, label=r"$j_4$")

    # Outside-R_a neighbour: position out of perception range.
    neighbour(0.85 * 1.2, 0.55 * 1.2, theta_deg=60, label=r"$j_\infty$",
              color="#666", alpha=0.55)

    # Radius labels.
    ax.annotate(r"$R_r$",
                xy=(R_r * np.cos(np.deg2rad(-50)),
                    R_r * np.sin(np.deg2rad(-50))),
                xytext=(0.66, -1.14), fontsize=14,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#444"))
    ax.annotate(r"$R_a$",
                xy=(R_a * np.cos(np.deg2rad(-30)),
                    R_a * np.sin(np.deg2rad(-30))),
                xytext=(1.26, -0.74), fontsize=14,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#444"))

    # Legend (zone colours).
    handles = [
        Patch(facecolor=rep_color, alpha=0.55, edgecolor="#9c3a3a",
              label=r"Repulsion ($d<R_r$)"),
        Patch(facecolor=ali_color, alpha=0.55, edgecolor="#3a4a78",
              label=r"Alignment ($R_r \leq d < R_a$)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=12,
              framealpha=0.92)

    ax.set_xlim(-1.86, 1.86)
    ax.set_ylim(-1.44, 1.44)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"(a) Two-zone Vicsek update "
                 r"(full $360^\circ$ vision)",
                 fontsize=14)

    fig.tight_layout()
    _save(fig, "fig_model_schematic.pdf")


FOCAL_RED = "#c83a3a"
REP_COLOR = "#e07b7b"
ALI_COLOR = "#9bb8de"

R_R_SCHEMA = 0.45 * 1.2   # zone radii used in panel (a)
R_A_SCHEMA = 0.7  * 1.2
V_VIS      = 0.40          # visualisation step: 8 x v_0 (= 0.05); the
                           # angular update is the standard rule, only
                           # the displacement is inflated for legibility


def _apply_rule(pos_i, theta_i, pos_j, theta_j, R_r, R_a):
    """Apply the model angular update to a focal particle i given
    its neighbours j (positions and headings as arrays). Returns
    the new heading theta_i_star (before noise). Implements exactly
    the _zonal_update logic of vicsek.py: repulsion > alignment >
    inertia, repulsion = atan2(-sum dy_ij/d, -sum dx_ij/d), alignment
    = atan2(sum sin th_j, sum cos th_j) over neighbours in R_r..R_a.
    """
    dx = pos_j[:, 0] - pos_i[0]
    dy = pos_j[:, 1] - pos_i[1]
    d  = np.hypot(dx, dy)
    rep_mask = (d > 0) & (d < R_r)
    ali_mask = (d >= R_r) & (d < R_a)
    if rep_mask.any():
        rx = -np.sum(dx[rep_mask] / d[rep_mask])
        ry = -np.sum(dy[rep_mask] / d[rep_mask])
        return float(np.arctan2(ry, rx))
    if ali_mask.any():
        sx = float(np.sum(np.sin(theta_j[ali_mask])))
        cx = float(np.sum(np.cos(theta_j[ali_mask])))
        return float(np.arctan2(sx, cx))
    return float(theta_i)


def _setup_rule_pair(title_main):
    """Two side-by-side subpanels labelled t and t+dt, same axis
    limits and aspect. Returns (fig, ax_t, ax_tp)."""
    fig, (ax_t, ax_tp) = plt.subplots(1, 2, figsize=(5.4, 4.4),
                                       sharex=True, sharey=True)
    for ax, lab in zip((ax_t, ax_tp), (r"$t$", r"$t + \delta t$")):
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.45, 1.55)
        ax.text(0.5, 1.02, lab, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=14,
                fontweight="bold")
    fig.suptitle(title_main, fontsize=13.5, y=0.97)
    return fig, ax_t, ax_tp


def _draw_zones(ax):
    """Draw the R_r repulsion disk and the R_r..R_a alignment
    annulus around the origin, same style as panel (a)."""
    from matplotlib.patches import Wedge
    rep = Wedge((0, 0), R_R_SCHEMA, 0, 360,
                facecolor=REP_COLOR, alpha=0.55,
                edgecolor="#9c3a3a", lw=0.9, zorder=1)
    ali = Wedge((0, 0), R_A_SCHEMA, 0, 360,
                width=R_A_SCHEMA - R_R_SCHEMA,
                facecolor=ALI_COLOR, alpha=0.55,
                edgecolor="#3a4a78", lw=0.9, zorder=1)
    ax.add_patch(rep); ax.add_patch(ali)


def _draw_particle(ax, pos, heading, *, color, scatter_size,
                   arrow_len, lw, label=None, label_offset=(0.07, 0.08),
                   label_fontsize=13, label_color=None, alpha=1.0):
    x, y = float(pos[0]), float(pos[1])
    tx = x + arrow_len * np.cos(heading)
    ty = y + arrow_len * np.sin(heading)
    ax.annotate("", xy=(tx, ty), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                alpha=alpha), zorder=6)
    ax.scatter([x], [y], s=scatter_size, color=color, alpha=alpha,
               edgecolor="white", lw=0.9, zorder=7)
    if label:
        if label_color is None:
            label_color = color
        ax.text(x + label_offset[0], y + label_offset[1], label,
                fontsize=label_fontsize, color=label_color,
                fontweight="bold", zorder=7)


def _evolve(pos_i, theta_i, pos_j, theta_j, R_r, R_a, v_vis):
    """Advance the whole configuration by one schematic step:
    angular update from _apply_rule for the focal i, identity
    update (no neighbours assumed) for each j_k, then ballistic
    displacement with step v_vis (visualisation scale) for every
    particle along its new heading.
    """
    new_th_i = _apply_rule(pos_i, theta_i, pos_j, theta_j, R_r, R_a)
    new_pos_i = pos_i + v_vis * np.array([np.cos(new_th_i),
                                          np.sin(new_th_i)])
    new_pos_j = pos_j.copy()
    new_th_j  = theta_j.copy()
    for k in range(len(theta_j)):
        new_pos_j[k] = pos_j[k] + v_vis * np.array(
            [np.cos(theta_j[k]), np.sin(theta_j[k])])
    return new_pos_i, new_th_i, new_pos_j, new_th_j


def _render_frame(ax, pos_i, theta_i, pos_j, theta_j, j_labels):
    """Render one frame (zones + focal + neighbours). Heading
    vectors are scaled up by ~55-60% relative to the panel-(a)
    defaults so they read as the dominant visual cue in the
    paired t / t+dt frames (focal arrow_len 0.288 -> 0.45,
    neighbour arrow_len 0.2016 -> 0.32; focal lw 3.1 -> 4.0,
    neighbour lw 1.92 -> 2.6)."""
    _draw_zones(ax)
    _draw_particle(ax, pos_i, theta_i, color=FOCAL_RED,
                   scatter_size=108, arrow_len=0.45, lw=4.0,
                   label=r"$i$", label_offset=(0.07, -0.30),
                   label_fontsize=15)
    for k, (pos, th, lab) in enumerate(zip(pos_j, theta_j, j_labels)):
        _draw_particle(ax, pos, th, color="#1f4ea1",
                       scatter_size=50, arrow_len=0.32, lw=2.6,
                       label=lab, label_offset=(0.06, 0.09),
                       label_fontsize=12)


def fig_rule_inertia():
    """Schematic (c): inertia. No neighbour inside R_a -> the focal
    heading is preserved and the position drifts ballistically.
    Standard params (R_r = 0.45, R_a = 0.7, v_0 = 0.05); displacement
    is rendered at v_vis = 0.40 (= 8 v_0) for visibility.
    """
    fig, ax_t, ax_tp = _setup_rule_pair(
        r"(c) Inertia: no neighbour in $R_a$"
        r" $\Rightarrow \vec e_i$ unchanged")

    pos_i = np.array([-0.55, -0.10])
    th_i  = np.deg2rad(20.0)
    pos_j = np.array([
        [ 1.05,  1.05],   # outside R_a (far up-right)
        [-1.20,  0.90],   # outside R_a (far up-left)
    ])
    th_j  = np.array([np.deg2rad(160.0), np.deg2rad(-20.0)])

    _render_frame(ax_t, pos_i, th_i, pos_j, th_j,
                  j_labels=[r"$j_1$", r"$j_2$"])

    new_pos_i, new_th_i, new_pos_j, new_th_j = _evolve(
        pos_i, th_i, pos_j, th_j, R_R_SCHEMA, R_A_SCHEMA, V_VIS)
    _render_frame(ax_tp, new_pos_i, new_th_i, new_pos_j, new_th_j,
                  j_labels=[r"$j_1$", r"$j_2$"])

    ax_tp.plot([pos_i[0], new_pos_i[0]],
               [pos_i[1], new_pos_i[1]],
               ls=":", color="#444", lw=0.9, zorder=2)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save(fig, "fig_rule_inertia.pdf")


def fig_rule_repulsion():
    """Schematic (d): repulsion. A neighbour j_1 sits inside R_r;
    the focal heading at t+dt points away from it
    (e_i^* = -hat(x_{j1} - x_i)). Standard params; v_vis = 0.40.
    """
    fig, ax_t, ax_tp = _setup_rule_pair(
        r"(d) Repulsion: $d_{ij_1}<R_r$"
        r" $\Rightarrow \vec e_i^{\,\star} = -\widehat{x_{j_1}-x_i}$")

    pos_i = np.array([0.0, 0.0])
    th_i  = np.deg2rad(45.0)
    pos_j = np.array([
        [-0.22,  0.27],   # inside R_r -> triggers repulsion
        [ 0.55, -0.25],   # inside alignment annulus (overridden)
        [ 1.10,  0.90],   # outside R_a
    ])
    th_j  = np.array([np.deg2rad(110.0),
                      np.deg2rad(  0.0),
                      np.deg2rad(-90.0)])

    _render_frame(ax_t, pos_i, th_i, pos_j, th_j,
                  j_labels=[r"$j_1$", r"$j_2$", r"$j_3$"])

    new_pos_i, new_th_i, new_pos_j, new_th_j = _evolve(
        pos_i, th_i, pos_j, th_j, R_R_SCHEMA, R_A_SCHEMA, V_VIS)
    _render_frame(ax_tp, new_pos_i, new_th_i, new_pos_j, new_th_j,
                  j_labels=[r"$j_1$", r"$j_2$", r"$j_3$"])

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save(fig, "fig_rule_repulsion.pdf")


def fig_rule_alignment():
    """Schematic (e): alignment. No neighbour in R_r; three
    neighbours sit in the alignment annulus, so the focal heading
    at t+dt is the circular mean of their headings. Standard params;
    v_vis = 0.40.
    """
    fig, ax_t, ax_tp = _setup_rule_pair(
        r"(e) Alignment: $R_r\!\leq\! d_{ij}\!<\!R_a$"
        r" $\Rightarrow \vec e_i^{\,\star} ="
        r" \mathrm{atan2}(\!\sum\!\sin\theta_j,\!\sum\!\cos\theta_j)$")

    pos_i = np.array([0.0, 0.0])
    th_i  = np.deg2rad(-30.0)
    pos_j = np.array([
        [ 0.55,  0.40],   # in alignment annulus
        [-0.58,  0.28],   # in alignment annulus
        [ 0.10, -0.62],   # in alignment annulus
        [ 1.20,  0.95],   # outside R_a (irrelevant)
    ])
    th_j  = np.array([np.deg2rad( 30.0),
                      np.deg2rad( 50.0),
                      np.deg2rad( 40.0),
                      np.deg2rad(180.0)])

    _render_frame(ax_t, pos_i, th_i, pos_j, th_j,
                  j_labels=[r"$j_1$", r"$j_2$", r"$j_3$", r"$j_4$"])

    new_pos_i, new_th_i, new_pos_j, new_th_j = _evolve(
        pos_i, th_i, pos_j, th_j, R_R_SCHEMA, R_A_SCHEMA, V_VIS)
    _render_frame(ax_tp, new_pos_i, new_th_i, new_pos_j, new_th_j,
                  j_labels=[r"$j_1$", r"$j_2$", r"$j_3$", r"$j_4$"])

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save(fig, "fig_rule_alignment.pdf")


def main():
    fig_noise_pdf()
    fig_model_schematic()
    fig_rule_inertia()
    fig_rule_repulsion()
    fig_rule_alignment()
    fig_snapshots()

    npz_3d = DATA / "sweep_3d.npz"
    if npz_3d.exists():
        fig_3d_phase(npz_3d)
    else:
        print(f"[warn] {npz_3d} not found -- run run_3d_sweep.py first")

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

    npz_diff = DATA / "diffusion.npz"
    npz_cl = DATA / "clusters.npz"
    if npz_diff.exists() and npz_cl.exists():
        fig_diffusion(npz_diff, npz_cl)
    else:
        print("[warn] diffusion.npz / clusters.npz not found -- "
              "run run_diffusion.py and run_clusters.py first")

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

if __name__ == "__main__":
    main()
