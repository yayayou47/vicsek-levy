"""
Builders for the remaining post-revision figures:
  - fig_hysteresis.pdf: original 3-alpha loops + slow-ramp overlay for
    alpha=1, 2 with the corrected loop areas.
  - fig_adaptive_pilot.pdf: 4-size FSS with per-seed bootstrap error
    bars on chi(eta), the FSS log-log slope panel, and the alpha_i
    statistics.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style


HERE = Path(__file__).resolve().parent
V1_ROOT = HERE.parent.parent if HERE.parent.name == "notes" else HERE.parent
DATA = V1_ROOT / "data"
FIG = V1_ROOT / "figures"
style.apply()


def _save(fig, name):
    out = FIG / name
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"saved: {out}")


def _block_smooth(y, bw=80):
    n = len(y) // bw
    return np.mean(y[: n * bw].reshape(n, bw), axis=1)


def fig_hysteresis():
    z_fast = np.load(DATA / "hysteresis.npz")
    z_slow = np.load(DATA / "hysteresis_slow.npz")
    alphas_fast = z_fast["alphas"]
    eta_fast = z_fast["eta_path"]; phi_fast = z_fast["phi_traj"]
    T_up_fast = int(z_fast["params"][6])
    alphas_slow = z_slow["alphas"]
    eta_slow = z_slow["eta_path"]; phi_slow = z_slow["phi_traj"]
    T_up_slow = int(z_slow["params"][6])

    # 4 panels: 3 alphas at fast ramp + 1 panel comparing fast vs slow.
    # The (e)..(h) panel letters continue the (a)-(d) sequence of
    # fig_fss.pdf (order parameter, chi, U_4, chi_max(L)) so the
    # composite Figure 3 caption reads in one continuous alphabet.
    fig, axes = plt.subplots(1, 4,
                             figsize=(style.DOUBLE_COL[0], 2.4),
                             sharey=True)
    panel_letters = ["(e)", "(f)", "(g)", "(h)"]
    for ia, ax in enumerate(axes[:3]):
        eup = eta_fast[:T_up_fast]; pup = phi_fast[ia, :T_up_fast]
        edn = eta_fast[T_up_fast:]; pdn = phi_fast[ia, T_up_fast:]
        ax.plot(_block_smooth(eup), _block_smooth(pup),
                "-", color="#1f4ea1", lw=1.3, label=r"ramp $\uparrow$")
        ax.plot(_block_smooth(edn), _block_smooth(pdn),
                "-", color="#d76f3a", lw=1.3, label=r"ramp $\downarrow$")
        eg = np.linspace(0, eta_fast.max(), 80)
        pi_up = np.interp(eg, _block_smooth(eup), _block_smooth(pup))
        pi_dn = np.interp(eg, _block_smooth(edn)[::-1],
                          _block_smooth(pdn)[::-1])
        _trapz = (getattr(np, "trapezoid", None)
                  or getattr(np, "trapz", None))
        area = float(_trapz(pi_up - pi_dn, eg))
        gap = float(np.max(pi_up - pi_dn))
        ax.set_title(fr"{panel_letters[ia]} $\alpha = "
                     fr"{alphas_fast[ia]:.1f}$"
                     "\n"
                     fr"area $= {area:.3f}$, gap $= {gap:.2f}$",
                     fontsize=7)
        ax.set_xlabel(r"noise $\eta$")
        if ia == 0:
            ax.set_ylabel(r"$\langle\varphi\rangle$")
            ax.legend(loc="upper right", fontsize=7)
        ax.set_ylim(-0.02, 1.05); ax.tick_params(labelsize=8)

    # 4th panel: slow ramp comparison at alpha = 2 and alpha = 1
    ax = axes[3]
    cols = {1.0: "#5c8c39", 2.0: "#3a4ba8"}
    for ia_s, a in enumerate(alphas_slow):
        eup = eta_slow[:T_up_slow]; pup = phi_slow[ia_s, :T_up_slow]
        edn = eta_slow[T_up_slow:]; pdn = phi_slow[ia_s, T_up_slow:]
        ax.plot(_block_smooth(eup), _block_smooth(pup),
                "-", color=cols[float(a)], lw=1.3,
                label=fr"$\alpha = {a:.0f}$ slow")
        ax.plot(_block_smooth(edn), _block_smooth(pdn),
                "--", color=cols[float(a)], lw=1.0)
    ax.set_title(r"(h) slow ramp $T = 32000$" "\n"
                 r"area: $\alpha{=}2{:}\,0.006$, $\alpha{=}1{:}\,0$",
                 fontsize=7)
    ax.set_xlabel(r"noise $\eta$")
    ax.set_ylim(-0.02, 1.05); ax.tick_params(labelsize=8)
    ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    _save(fig, "fig_hysteresis.pdf")


def _bootstrap_slope(Ls, chi_max_per_seed, n_boot=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    n_L, n_s = chi_max_per_seed.shape
    slopes = []
    chis = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_s, size=n_s)
        cb = chi_max_per_seed[:, idx].mean(axis=1)
        if (cb > 0).all():
            slopes.append(np.polyfit(np.log(Ls), np.log(cb), 1)[0])
            chis.append(cb)
    slopes = np.array(slopes); chis = np.array(chis)
    return float(slopes.mean()), float(slopes.std()), \
        chis.mean(axis=0), chis.std(axis=0)


def fig_adaptive_pilot():
    z = np.load(DATA / "adaptive_perseed.npz", allow_pickle=True)
    Ls = z["Ls"]; modes = z["modes"]; etas = z["etas"]
    phi = z["phi"]; chi = z["chi"]; U4 = z["U4"]
    am = z["alpha_mean"]; as_ = z["alpha_std"]

    # Harmonised 7-size lever: merge the L = 64, 91, 128 extension.
    ext_path = DATA / "adaptive_perseed_ext.npz"
    if ext_path.exists():
        zE = np.load(ext_path, allow_pickle=True)
        assert list(zE["modes"]) == list(modes), "mode mismatch"
        # both arrays share the (mode, L, eta, seed) layout; concat
        # along the L axis after intersecting seed and eta counts.
        n_seed = min(phi.shape[-1], zE["phi"].shape[-1])
        n_eta = min(phi.shape[2], zE["phi"].shape[2])
        phi = np.concatenate([phi[..., :n_eta, :n_seed],
                              zE["phi"][..., :n_eta, :n_seed]], axis=1)
        chi = np.concatenate([chi[..., :n_eta, :n_seed],
                              zE["chi"][..., :n_eta, :n_seed]], axis=1)
        U4 = np.concatenate([U4[..., :n_eta, :n_seed],
                             zE["U4"][..., :n_eta, :n_seed]], axis=1)
        am = np.concatenate([am[..., :n_eta, :n_seed],
                             zE["alpha_mean"][..., :n_eta, :n_seed]], axis=1)
        as_ = np.concatenate([as_[..., :n_eta, :n_seed],
                              zE["alpha_std"][..., :n_eta, :n_seed]], axis=1)
        Ls = np.concatenate([Ls, zE["Ls"]])

    fig = plt.figure(figsize=(style.DOUBLE_COL[0], 5.0))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.32,
                          height_ratios=[1.2, 1.0])
    cmap = plt.get_cmap("plasma")
    n_L = len(Ls)
    cols = [cmap(0.15 + 0.7 * i / max(1, n_L - 1)) for i in range(n_L)]

    # Top row: phi(eta), chi(eta), U4(eta) for the adaptive run with
    # per-L curves and 1-sigma bands; fixed-Cauchy as dashed reference
    ax_p = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[0, 2])
    for iL, L in enumerate(Ls):
        # adaptive run
        ph = phi[1, iL]; ch = chi[1, iL]; u4 = U4[1, iL]
        for ax, dat in [(ax_p, ph), (ax_c, ch), (ax_u, u4)]:
            m = dat.mean(axis=-1); s = dat.std(axis=-1)
            ax.plot(etas, m, "o-", color=cols[iL], ms=3, lw=0.9,
                    label=fr"$L={int(L)}$" if ax is ax_p else None)
            ax.plot(etas, m - s, ls="--", color=cols[iL], lw=0.6,
                    alpha=0.7)
            ax.plot(etas, m + s, ls="--", color=cols[iL], lw=0.6,
                    alpha=0.7)
        # fixed-Cauchy reference (dashed, thin, gray)
        if iL == len(Ls) - 1:
            for ax, idx in [(ax_p, 0), (ax_c, 1), (ax_u, 2)]:
                d = [phi, chi, U4][idx][0, iL]
                m = d.mean(axis=-1)
                ax.plot(etas, m, ":", color="#666", lw=0.8,
                        label="fixed-Cauchy ref" if idx == 0 else None)
    ax_p.set_ylabel(r"$\langle\varphi\rangle$"); ax_p.set_ylim(-0.03, 1.05)
    ax_c.set_ylabel(r"$\chi$"); ax_u.set_ylabel(r"$U_4$")
    for ax in (ax_p, ax_c, ax_u):
        ax.set_xlabel(r"noise $\eta$"); ax.tick_params(labelsize=8)
    ax_p.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)

    # Bottom row: alpha statistics, then chi_max(L) log-log
    ax_a = fig.add_subplot(gs[1, 0])
    ax_as = fig.add_subplot(gs[1, 1])
    ax_fss = fig.add_subplot(gs[1, 2])

    # Mean and std of alpha_i across L's: show eta dependence
    for iL, L in enumerate(Ls):
        ax_a.plot(etas, am[1, iL].mean(axis=-1), "o-", color=cols[iL],
                  ms=3, lw=0.9, label=fr"$L={int(L)}$" if iL == 0 else None)
        ax_as.plot(etas, as_[1, iL].mean(axis=-1), "s-", color=cols[iL],
                   ms=3, lw=0.9)
    ax_a.set_xlabel(r"noise $\eta$")
    ax_a.set_ylabel(r"$\langle\alpha_i\rangle$")
    ax_as.set_xlabel(r"noise $\eta$")
    ax_as.set_ylabel(r"$\sigma(\alpha_i)$")
    ax_as.axhline(0.33, ls=":", c="grey", lw=0.7)
    ax_a.tick_params(labelsize=8); ax_as.tick_params(labelsize=8)

    # FSS log-log fit panel: chi_max(L) per mode with bootstrap.
    # Seed 5 matches tools/update_chiffres.py group 5, so the
    # printed slopes coincide with the numbers in Sec. III F.
    rng = np.random.default_rng(5)
    mode_cols = {"fixed": "#666", "adaptive": "#d76f3a"}
    for im, lbl in enumerate(modes):
        cmax = chi[im, :, :, :].max(axis=1)  # (L, seed)
        s, se, c_m, c_s = _bootstrap_slope(Ls, cmax, rng=rng)
        col = mode_cols[str(lbl)]
        ax_fss.errorbar(Ls, c_m, yerr=c_s, fmt="o", color=col,
                        ms=5, capsize=3, lw=1.0,
                        label=fr"{lbl}: $L^{{{s:.2f} \pm {se:.2f}}}$")
        Lf = np.linspace(Ls[0]*0.9, Ls[-1]*1.1, 50)
        b = np.polyfit(np.log(Ls), np.log(c_m), 1)
        ax_fss.plot(Lf, np.exp(b[1])*Lf**b[0], "--",
                    color=col, lw=0.8)
    ax_fss.set_xscale("log"); ax_fss.set_yscale("log")
    ax_fss.set_xticks(list(Ls))
    ax_fss.set_xticklabels([str(int(L)) for L in Ls])
    ax_fss.minorticks_off()
    ax_fss.set_xlabel(r"$L$"); ax_fss.set_ylabel(r"$\chi_{\max}$")
    ax_fss.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax_fss.tick_params(labelsize=8)

    fig.tight_layout()
    _save(fig, "fig_adaptive_pilot.pdf")


def _add_zoom_inset(ax, x_all, y_all, th_all, L, cmap,
                    src_frac=(0.70, 0.70)):
    """Circular 5x zoom inset in the top-right corner of ax.

    Geometry. The display circle has diameter L/2 in panel
    coordinates (the inset takes up half the panel side). The
    source patch is a disc of diameter L/10 in real coordinates
    (strict 5x linear magnification), centred in the top-right
    quadrant at (src_frac[0]*L, src_frac[1]*L). The same source
    patch is used across panels so the local arrow configuration
    is directly comparable.

    Arrow style. Each particle is drawn as a small filled dot at
    its position, a thin coloured stroke towards its heading,
    and a tiny arrowhead at the tip --- the dot+stroke+arrowhead
    triplet stays legible even when individual arrows are small,
    which is the regime here (~100--200 particles in the disc)."""
    from matplotlib.patches import Circle

    side_frac = 0.50    # inset axes side as fraction of panel
    pad = 0.02
    inset_bl = (1 - side_frac - pad, 1 - side_frac - pad)
    ax_in = ax.inset_axes([*inset_bl, side_frac, side_frac])

    cx = src_frac[0] * L
    cy = src_frac[1] * L
    real_half = L / 20.0    # source diameter = L/10 (5x zoom)

    in_disc = ((x_all - cx) ** 2 + (y_all - cy) ** 2
               <= real_half ** 2)
    x = x_all[in_disc]; y = y_all[in_disc]; th = th_all[in_disc]

    ax_in.set_xlim(cx - real_half, cx + real_half)
    ax_in.set_ylim(cy - real_half, cy + real_half)
    ax_in.set_aspect("equal")
    ax_in.set_xticks([]); ax_in.set_yticks([])
    ax_in.patch.set_visible(False)
    for spine in ax_in.spines.values():
        spine.set_visible(False)

    bg = Circle((0.5, 0.5), 0.5, transform=ax_in.transAxes,
                facecolor="white", edgecolor="none", zorder=0)
    ax_in.add_patch(bg)

    if x.size > 0:
        from matplotlib.patches import FancyArrowPatch
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        col_val = (th + np.pi) / (2 * np.pi)

        clip = Circle((0.5, 0.5), 0.5,
                      transform=ax_in.transAxes,
                      facecolor="none", edgecolor="none")
        ax_in.add_patch(clip)

        # Particles in the zoom area shrunk 50% in linear size:
        # arrow length 13% -> 6.5% of source-disc diameter,
        # shaft and head proportionally thinner, dot area
        # quartered so the dot diameter is halved.
        arr_len = (2.0 * real_half) * 0.065

        sc = ax_in.scatter(x, y, s=0.6, c=col_val, cmap=cmap_obj,
                           vmin=0.0, vmax=1.0,
                           edgecolors="white", linewidths=0.08,
                           zorder=4)
        sc.set_clip_path(clip)

        # Per-particle arrow with filled triangular head, same
        # arrowstyle ("-|>") as the Fig. 1 neighbour vectors.
        # shrinkA = shrinkB = 0 so the shaft starts exactly at
        # the tail dot (no gap) and the arrowhead tip lands at
        # the prescribed endpoint.
        for xi, yi, thi, cv in zip(x, y, th, col_val):
            colour = cmap_obj(cv)
            arrow = FancyArrowPatch(
                (xi, yi),
                (xi + arr_len * np.cos(thi),
                 yi + arr_len * np.sin(thi)),
                arrowstyle="-|>", color=colour,
                lw=0.225, mutation_scale=2.0,
                shrinkA=0.0, shrinkB=0.0,
                zorder=3)
            ax_in.add_patch(arrow)
            arrow.set_clip_path(clip)

    outline = Circle((0.5, 0.5), 0.5, transform=ax_in.transAxes,
                     fill=False, edgecolor="black",
                     linewidth=0.9, zorder=10)
    ax_in.add_patch(outline)


def fig_order_snapshots():
    """2x3 grid of particle snapshots illustrating §order:
    rows are alpha = 1 (Cauchy) and alpha = 2 (Gaussian); columns
    are ordered (eta << eta_c), near-critical (eta ~ eta_c), and
    disordered (eta >> eta_c). Arrows are coloured by heading
    angle via the cyclic 'twilight' colormap so collective
    alignment shows up as colour uniformity. A 10x circular zoom
    inset in each panel's top-right corner makes the local
    alignment visible at the per-arrow scale."""
    z = np.load(DATA / "order_snapshots.npz")
    alphas = z["alphas"]
    etas = z["etas"]
    snap_x = z["snap_x"]; snap_y = z["snap_y"]
    snap_theta = z["snap_theta"]
    phi = z["phi_final"]
    params = z["params"]
    L = float(params[1])

    regimes = ["ordered", "near-critical", "disordered"]
    fig, axes = plt.subplots(2, 3,
                             figsize=(style.DOUBLE_COL[0], 4.6),
                             sharex=True, sharey=True)
    arr_scale = 28.0
    cmap = plt.get_cmap("twilight")

    # Subsample arrows: ~1500 per panel so the inter-particle
    # spacing is ~2.3 data units at L = 91, large enough that
    # the main-panel glyphs stay individually readable.
    rng = np.random.default_rng(0)
    n_show = min(int(round(0.18 * L * L)), 1800)

    # Arrow length in data units. The main-panel dot+shaft+head
    # glyph is drawn 30% larger and thicker than the zoom-inset
    # glyph so it stays legible at the full-box field of view.
    arr_len = 2.08 * (L / 91.0)

    for ic in range(6):
        ia = ic // 3
        ir = ic % 3
        ax = axes[1 - ia, ir]
        N = snap_x[ic].shape[0]
        idx = rng.choice(N, size=min(n_show, N), replace=False)
        x = snap_x[ic][idx]
        y = snap_y[ic][idx]
        th = snap_theta[ic][idx]
        col_val = (th + np.pi) / (2 * np.pi)

        # Each main-panel particle uses the same dot/shaft/head
        # glyph as the zoom inset, scaled up 30% in size and
        # stroke width so it stays legible at the full-box field
        # of view; the inset keeps the unscaled glyph on its
        # L/10 disc.
        from matplotlib.patches import FancyArrowPatch
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        ax.scatter(x, y, s=0.78, c=col_val, cmap=cmap_obj,
                   vmin=0.0, vmax=1.0,
                   edgecolors="white", linewidths=0.104, zorder=3)
        for xi, yi, thi, cv in zip(x, y, th, col_val):
            colour = cmap_obj(cv)
            a = FancyArrowPatch(
                (xi, yi),
                (xi + arr_len * np.cos(thi),
                 yi + arr_len * np.sin(thi)),
                arrowstyle="-|>", color=colour,
                lw=0.2925, mutation_scale=2.6,
                shrinkA=0.0, shrinkB=0.0,
                zorder=2)
            ax.add_patch(a)
        _add_zoom_inset(ax, snap_x[ic], snap_y[ic], snap_theta[ic],
                        L, cmap)
        ax.set_xlim(0, L); ax.set_ylim(0, L)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        letter = chr(ord('a') + (1 - ia) * 3 + ir)
        ax.set_title(f"({letter}) $\\eta = {etas[ic]:.2f}$, "
                     f"$\\langle\\varphi\\rangle = {phi[ic]:.2f}$",
                     fontsize=9)
        if ir == 0:
            ax.set_ylabel(f"$\\alpha = {alphas[ic]:.0f}$",
                          fontsize=10, fontweight="bold")

    for ir, lab in enumerate(regimes):
        axes[0, ir].annotate(lab,
                             xy=(0.5, 1.22),
                             xycoords="axes fraction",
                             ha="center", va="bottom",
                             fontsize=10, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_order_snapshots.pdf")


if __name__ == "__main__":
    fig_hysteresis()
    fig_adaptive_pilot()
    fig_order_snapshots()
