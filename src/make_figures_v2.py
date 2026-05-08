"""
Updated figure builders for the major-revision data:
  - fig_fss.pdf rebuilt with 5 sizes (incl. L=64) and seed-bootstrap
    error bars; bottom row replaced by a chi_max(L) log-log fit panel
    per alpha that reports the slope with bootstrap SE.
  - fig_calibrated.pdf rebuilt to add the FSS-at-fixed-V chi_max(L)
    panel from the v2 calibrated sweep.
Reads:
  data/fss_perseed.npz, data/fss_L64.npz, data/fss_calibrated_v2.npz
Writes:
  figures/fig_fss.pdf, figures/fig_calibrated.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIG = HERE.parent / "figures"
style.apply()


def _save(fig, name):
    out = FIG / name
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"saved: {out}")


def _bootstrap_slope(Ls, chi_max_per_seed, n_boot=2000, rng=None):
    """chi_max_per_seed shape (n_L, n_seeds). Return (mean_slope,
    se_slope, mean_chi_per_L, se_chi_per_L)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_L, n_s = chi_max_per_seed.shape
    slopes = []
    chis_boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_s, size=n_s)
        chi_boot = chi_max_per_seed[:, idx].mean(axis=1)
        if (chi_boot > 0).all():
            slope = np.polyfit(np.log(Ls), np.log(chi_boot), 1)[0]
            slopes.append(slope)
            chis_boot.append(chi_boot)
    slopes = np.array(slopes)
    chis_boot = np.array(chis_boot)  # (n_boot, n_L)
    return (
        float(slopes.mean()), float(slopes.std()),
        chis_boot.mean(axis=0), chis_boot.std(axis=0),
    )


def fig_fss():
    """Top: phi(eta), chi(eta), U4(eta) at L in {15, 22, 30, 45, 64}
    with shaded 1-sigma seed-bootstrap bands (eta grid coarsened to
    where peaks live). Bottom: chi_max(L) log-log fit for alpha=1, 2."""
    z4 = np.load(DATA / "fss_perseed.npz")
    z64 = np.load(DATA / "fss_L64.npz")

    Ls4 = z4["Ls"]
    alphas = z4["alphas"]
    etas4 = z4["etas"]
    phi4 = z4["phi"]
    chi4 = z4["chi"]
    U4_4 = z4["U4"]

    L64 = float(z64["L"])
    etas64 = z64["etas"]
    phi64 = z64["phi"]    # (alpha, eta, seed)
    chi64 = z64["chi"]
    U4_64 = z64["U4"]

    Ls5 = np.concatenate([Ls4, [L64]])

    fig = plt.figure(figsize=(style.DOUBLE_COL[0], 5.6))
    # 2 rows x 3 cols, but bottom row only spans 2 panels (cols 1-2)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3,
                          height_ratios=[1.4, 1.0])

    cmap = plt.get_cmap("plasma")
    cols5 = [cmap(0.15 + 0.7 * i / 4) for i in range(5)]

    # === Top row: alpha=1 ===
    ax_phi = fig.add_subplot(gs[0, 0])
    ax_chi = fig.add_subplot(gs[0, 1])
    ax_U4 = fig.add_subplot(gs[0, 2])
    ia = 0  # alpha = 1
    for iL, L in enumerate(Ls5):
        if iL < 4:
            ph = phi4[iL, ia]; ch = chi4[iL, ia]; u4 = U4_4[iL, ia]
            etas = etas4
        else:
            ph = phi64[ia]; ch = chi64[ia]; u4 = U4_64[ia]
            etas = etas64
        ph_m = ph.mean(axis=-1); ph_s = ph.std(axis=-1)
        ch_m = ch.mean(axis=-1); ch_s = ch.std(axis=-1)
        u4_m = u4.mean(axis=-1); u4_s = u4.std(axis=-1)
        lbl = fr"$L={int(L)}$"
        ax_phi.plot(etas, ph_m, "o-", color=cols5[iL], ms=3, lw=0.8, label=lbl)
        ax_phi.fill_between(etas, ph_m - ph_s, ph_m + ph_s,
                            color=cols5[iL], alpha=0.18, lw=0)
        ax_chi.plot(etas, ch_m, "s-", color=cols5[iL], ms=3, lw=0.8)
        ax_chi.fill_between(etas, np.maximum(ch_m - ch_s, 1e-3),
                            ch_m + ch_s, color=cols5[iL], alpha=0.18, lw=0)
        ax_U4.plot(etas, u4_m, "d-", color=cols5[iL], ms=3, lw=0.8)
        ax_U4.fill_between(etas, u4_m - u4_s, u4_m + u4_s,
                           color=cols5[iL], alpha=0.18, lw=0)
    ax_phi.set_ylim(-0.03, 1.05)
    ax_phi.set_ylabel(r"$\langle\varphi\rangle$")
    ax_chi.set_ylabel(r"$\chi$")
    ax_U4.set_ylabel(r"$U_4$")
    ax_U4.axhline(2.0/3.0, ls=":", c="grey", lw=0.7)
    for ax in (ax_phi, ax_chi, ax_U4):
        ax.set_xlabel(r"noise $\eta$")
        ax.tick_params(labelsize=7)
    ax_phi.text(0.04, 0.92, r"$\alpha = 1$ (Cauchy)",
                transform=ax_phi.transAxes,
                fontsize=8, fontweight="bold")
    ax_phi.legend(loc="upper right", fontsize=6, framealpha=0.9, ncol=2)

    # === Bottom row: chi_max(L) log-log fit per alpha ===
    ax_a1 = fig.add_subplot(gs[1, 0])
    ax_a2 = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(0)
    for iax, (ax, ia, alabel) in enumerate(
        [(ax_a1, 0, r"$\alpha = 1$"), (ax_a2, 1, r"$\alpha = 2$")]
    ):
        chi_max_per_L_seed = np.zeros((5, 5))
        for iL in range(4):
            chi_max_per_L_seed[iL] = chi4[iL, ia, :, :].max(axis=0)
        chi_max_per_L_seed[4] = chi64[ia, :, :].max(axis=0)
        s, se, chi_m, chi_s = _bootstrap_slope(Ls5, chi_max_per_L_seed,
                                                rng=rng)
        ax.errorbar(Ls5, chi_m, yerr=chi_s, fmt="o", color=cols5[2],
                    ms=5, capsize=3, lw=1.0)
        Lf = np.linspace(Ls5[0]*0.9, Ls5[-1]*1.1, 50)
        b = np.polyfit(np.log(Ls5), np.log(chi_m), 1)
        ax.plot(Lf, np.exp(b[1]) * Lf**b[0], "k--", lw=0.8,
                label=fr"$\propto L^{{{s:.2f} \pm {se:.2f}}}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(list(Ls5))
        ax.set_xticklabels([str(int(L)) for L in Ls5])
        ax.minorticks_off()
        ax.set_xlabel(r"$L$")
        ax.set_ylabel(r"$\chi_{\max}$")
        ax.legend(loc="lower right", fontsize=8)
        ax.text(0.04, 0.92, alabel, transform=ax.transAxes,
                fontsize=8, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig_fss.pdf")


def fig_fss_alpha2():
    """A version with the alpha=2 row in the top instead of alpha=1.
    For when the manuscript wants the canonical first-order signature
    in the lead. Saves to fig_fss_a2.pdf."""
    z4 = np.load(DATA / "fss_perseed.npz")
    z64 = np.load(DATA / "fss_L64.npz")
    alphas = z4["alphas"]
    Ls4 = z4["Ls"]; etas4 = z4["etas"]
    phi4 = z4["phi"]; chi4 = z4["chi"]; U4_4 = z4["U4"]
    L64 = float(z64["L"]); etas64 = z64["etas"]
    phi64 = z64["phi"]; chi64 = z64["chi"]; U4_64 = z64["U4"]
    Ls5 = np.concatenate([Ls4, [L64]])

    fig = plt.figure(figsize=(style.DOUBLE_COL[0], 6.4))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3,
                          height_ratios=[1.0, 1.0, 1.0])
    cmap = plt.get_cmap("plasma")
    cols5 = [cmap(0.15 + 0.7 * i / 4) for i in range(5)]
    rng = np.random.default_rng(0)

    for irow, ia in enumerate([1, 0]):  # alpha=2 first, then alpha=1
        ax_phi = fig.add_subplot(gs[irow, 0])
        ax_chi = fig.add_subplot(gs[irow, 1])
        ax_U4 = fig.add_subplot(gs[irow, 2])
        for iL, L in enumerate(Ls5):
            if iL < 4:
                ph = phi4[iL, ia]; ch = chi4[iL, ia]; u4 = U4_4[iL, ia]
                etas = etas4
            else:
                ph = phi64[ia]; ch = chi64[ia]; u4 = U4_64[ia]
                etas = etas64
            ph_m = ph.mean(axis=-1); ph_s = ph.std(axis=-1)
            ch_m = ch.mean(axis=-1); ch_s = ch.std(axis=-1)
            u4_m = u4.mean(axis=-1); u4_s = u4.std(axis=-1)
            lbl = fr"$L={int(L)}$" if irow == 0 else None
            ax_phi.plot(etas, ph_m, "o-", color=cols5[iL], ms=3, lw=0.8,
                        label=lbl)
            ax_phi.fill_between(etas, ph_m-ph_s, ph_m+ph_s,
                                color=cols5[iL], alpha=0.18, lw=0)
            ax_chi.plot(etas, ch_m, "s-", color=cols5[iL], ms=3, lw=0.8)
            ax_chi.fill_between(etas, np.maximum(ch_m-ch_s, 1e-3), ch_m+ch_s,
                                color=cols5[iL], alpha=0.18, lw=0)
            ax_U4.plot(etas, u4_m, "d-", color=cols5[iL], ms=3, lw=0.8)
            ax_U4.fill_between(etas, u4_m-u4_s, u4_m+u4_s,
                               color=cols5[iL], alpha=0.18, lw=0)
        ax_phi.set_ylim(-0.03, 1.05)
        ax_phi.set_ylabel(r"$\langle\varphi\rangle$")
        ax_chi.set_ylabel(r"$\chi$")
        ax_U4.set_ylabel(r"$U_4$")
        ax_U4.axhline(2.0/3.0, ls=":", c="grey", lw=0.7)
        for ax in (ax_phi, ax_chi, ax_U4):
            ax.tick_params(labelsize=7)
            if irow == 1:
                ax.set_xlabel(r"noise $\eta$")
        alabel = fr"$\alpha = {alphas[ia]:.1f}$" + (
            " (Gauss.)" if ia == 1 else " (Cauchy)"
        )
        ax_phi.text(0.04, 0.92, alabel, transform=ax_phi.transAxes,
                    fontsize=8, fontweight="bold")
        if irow == 0:
            ax_phi.legend(loc="upper right", fontsize=6, ncol=2,
                          framealpha=0.9)

    # FSS log-log row
    ax_a1 = fig.add_subplot(gs[2, 0])
    ax_a2 = fig.add_subplot(gs[2, 1])
    for ax, ia, alabel in [
        (ax_a1, 1, r"$\alpha = 2$"), (ax_a2, 0, r"$\alpha = 1$"),
    ]:
        chi_max_per_L_seed = np.zeros((5, 5))
        for iL in range(4):
            chi_max_per_L_seed[iL] = chi4[iL, ia, :, :].max(axis=0)
        chi_max_per_L_seed[4] = chi64[ia, :, :].max(axis=0)
        s, se, chi_m, chi_s = _bootstrap_slope(Ls5, chi_max_per_L_seed,
                                                rng=rng)
        ax.errorbar(Ls5, chi_m, yerr=chi_s, fmt="o", color=cols5[2],
                    ms=5, capsize=3, lw=1.0)
        Lf = np.linspace(Ls5[0]*0.9, Ls5[-1]*1.1, 50)
        b = np.polyfit(np.log(Ls5), np.log(chi_m), 1)
        ax.plot(Lf, np.exp(b[1]) * Lf**b[0], "k--", lw=0.8,
                label=fr"$\propto L^{{{s:.2f} \pm {se:.2f}}}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(list(Ls5))
        ax.set_xticklabels([str(int(L)) for L in Ls5])
        ax.minorticks_off()
        ax.set_xlabel(r"$L$"); ax.set_ylabel(r"$\chi_{\max}$")
        ax.legend(loc="lower right", fontsize=8)
        ax.text(0.04, 0.92, alabel, transform=ax.transAxes,
                fontsize=8, fontweight="bold")
    # Hide the unused panel
    fig.add_subplot(gs[2, 2]).axis("off")

    fig.tight_layout()
    _save(fig, "fig_fss.pdf")


def fig_calibrated():
    """Calibrated noise: eta_alpha(V), <phi>(V), chi(V) at L=15, plus a
    new chi_max(L) panel at fixed V for alpha=1, 2 from the v2 sweep."""
    z = np.load(DATA / "calibrated_sweep.npz")
    alphas = z["alphas"]
    V_grid = z["V_grid"]
    eta_table = z["eta_table"]   # (n_alpha_old, n_V)
    phi_o = z["phi_mean"]
    chi_o = z["chi"]

    # v2 multi-L calibrated FSS
    z2 = np.load(DATA / "fss_calibrated_v2.npz")
    Ls = z2["Ls"]; alphas2 = z2["alphas"]; V2 = z2["V_targets"]
    eta_table2 = z2["eta_table"]
    chi2 = z2["chi"]   # (L, alpha, V, seed)

    fig, axes = plt.subplots(1, 4, figsize=(style.DOUBLE_COL[0], 2.5))
    cmap = plt.get_cmap("viridis")
    cols = [cmap(0.15 + 0.7 * i / max(1, len(alphas)-1))
            for i in range(len(alphas))]

    ax_a, ax_phi, ax_chi, ax_fss = axes
    for ia, a in enumerate(alphas):
        ax_a.plot(V_grid, eta_table[ia], "o-", color=cols[ia], ms=3,
                  lw=0.8, label=fr"$\alpha={a:.1f}$")
        ax_phi.plot(V_grid, phi_o[ia], "o-", color=cols[ia], ms=3, lw=0.8)
        ax_chi.plot(V_grid, chi_o[ia], "s-", color=cols[ia], ms=3, lw=0.8)
    ax_a.set_xlabel(r"calibrated $V$"); ax_a.set_ylabel(r"$\eta_\alpha(V)$")
    ax_phi.set_xlabel(r"$V$"); ax_phi.set_ylabel(r"$\langle\varphi\rangle$")
    ax_chi.set_xlabel(r"$V$"); ax_chi.set_ylabel(r"$\chi$")
    ax_a.legend(fontsize=7, loc="upper left", framealpha=0.9)
    for ax in (ax_a, ax_phi, ax_chi):
        ax.tick_params(labelsize=7)
    ax_a.set_title("(a) calibration", fontsize=8)
    ax_phi.set_title("(b) order vs V (L=15)", fontsize=8)
    ax_chi.set_title("(c) susceptibility vs V (L=15)", fontsize=8)

    # Panel (d): chi_max(L) at fixed V for alpha = 1, 2
    rng = np.random.default_rng(0)
    pmap = plt.get_cmap("plasma")
    a_cols = {1.0: pmap(0.25), 2.0: pmap(0.65)}
    for ia, a in enumerate(alphas2):
        chi_max_per_L_seed = chi2[:, ia, :, :].max(axis=1)  # (L, seed)
        s, se, chi_m, chi_s = _bootstrap_slope(Ls, chi_max_per_L_seed,
                                                rng=rng)
        ax_fss.errorbar(Ls, chi_m, yerr=chi_s, fmt="o",
                        color=a_cols[float(a)],
                        ms=5, capsize=3, lw=1.0,
                        label=fr"$\alpha = {a:.0f}$: $L^{{{s:.2f}\pm{se:.2f}}}$")
        Lf = np.linspace(Ls[0]*0.9, Ls[-1]*1.1, 50)
        b = np.polyfit(np.log(Ls), np.log(chi_m), 1)
        ax_fss.plot(Lf, np.exp(b[1])*Lf**b[0], "--",
                    color=a_cols[float(a)], lw=0.8)
    ax_fss.set_xscale("log"); ax_fss.set_yscale("log")
    ax_fss.set_xticks(list(Ls))
    ax_fss.set_xticklabels([str(int(L)) for L in Ls])
    ax_fss.minorticks_off()
    ax_fss.set_xlabel(r"$L$")
    ax_fss.set_ylabel(r"$\chi_{\max}$ at fixed $V$")
    ax_fss.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax_fss.tick_params(labelsize=7)
    ax_fss.set_title("(d) FSS at fixed V", fontsize=8)

    fig.tight_layout()
    _save(fig, "fig_calibrated.pdf")


if __name__ == "__main__":
    fig_fss_alpha2()
    fig_calibrated()
