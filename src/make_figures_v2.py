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
# After the v1 reorganization (commit f9754b2), the script lives at
# version1/notes/src/ while data and figures stay at the version1 root.
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
    """Single-row 1x4 layout on the 7-size lever L in
    {15, 22, 30, 45, 64, 91, 128}. Panels read alphabetically:

      (a) <phi>(eta) per L; alpha = 1 solid (with 1-sigma seed
          envelope filled), alpha = 2 overlaid as dashed mean.
      (b) chi(eta) per L, same convention.
      (c) U_4(eta) per L, same convention.
      (d) chi_max(L) log-log, bootstrap slope per alpha.

    Data loader matches tools/update_chiffres.py exactly: the
    fss_L{15..64}.warm20000.npz files (n_warm = 20000) are
    preferred over the pilot fss_perseed.npz / fss_L64.npz
    (n_warm in {1500, 2000}), so the bootstrap slopes printed in
    panel (d) coincide with the numbers quoted in the manuscript.

    The "alpha=1 solid / alpha=2 dashed" key is also written in
    the subtitles of (a)-(c) so the figure reads stand-alone.
    """
    levers = []
    for L_val in (15, 22, 30, 45, 64):
        z = np.load(DATA / f"fss_L{L_val}.warm20000.npz")
        levers.append({
            "L": float(z["L"]), "etas": z["etas"],
            "phi": z["phi"], "chi": z["chi"], "U4": z["U4"],
            "alphas": z["alphas"],
        })
    for fname in ("fss_L91.npz", "fss_L128.npz"):
        z = np.load(DATA / fname)
        levers.append({
            "L": float(z["L"]), "etas": z["etas"],
            "phi": z["phi"], "chi": z["chi"], "U4": z["U4"],
            "alphas": z["alphas"],
        })
    Ls7 = np.array([lev["L"] for lev in levers])
    alphas = levers[0]["alphas"]

    fig = plt.figure(figsize=(style.DOUBLE_COL[0] * 1.5, 2.4))
    gs = fig.add_gridspec(1, 4, wspace=0.40)

    cmap = plt.get_cmap("plasma")
    n_L = len(Ls7)
    cols7 = [cmap(0.15 + 0.7 * i / (n_L - 1)) for i in range(n_L)]
    A_COL = {1.0: "#d76f3a", 2.0: "#1f4ea1"}   # orange / blue

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_u = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[0, 3])

    # Panels (a)-(c): <phi>(eta), chi(eta), U_4(eta) per L,
    # alpha=1 solid with 1-sigma seed envelope filled, alpha=2
    # mean overlaid as dashed.
    for iL, lev in enumerate(levers):
        etas = lev["etas"]
        ph1 = lev["phi"][0]; ch1 = lev["chi"][0]; u41 = lev["U4"][0]
        ph2 = lev["phi"][1]; ch2 = lev["chi"][1]; u42 = lev["U4"][1]
        col = cols7[iL]
        lbl = fr"$L={int(lev['L'])}$"
        ph_m = ph1.mean(axis=-1); ph_s = ph1.std(axis=-1)
        ch_m = ch1.mean(axis=-1); ch_s = ch1.std(axis=-1)
        u4_m = u41.mean(axis=-1); u4_s = u41.std(axis=-1)
        ax_a.fill_between(etas, ph_m - ph_s, ph_m + ph_s,
                          color=col, alpha=0.18, lw=0)
        ax_a.plot(etas, ph_m, "-", color=col, lw=1.1, label=lbl)
        ax_a.plot(etas, ph2.mean(axis=-1), ls="--", color=col,
                  lw=0.9, alpha=0.85)
        ax_b.fill_between(etas, np.maximum(ch_m - ch_s, 1e-3),
                          ch_m + ch_s,
                          color=col, alpha=0.18, lw=0)
        ax_b.plot(etas, ch_m, "-", color=col, lw=1.1)
        ax_b.plot(etas, ch2.mean(axis=-1), ls="--", color=col,
                  lw=0.9, alpha=0.85)
        ax_u.fill_between(etas, u4_m - u4_s, u4_m + u4_s,
                          color=col, alpha=0.18, lw=0)
        ax_u.plot(etas, u4_m, "-", color=col, lw=1.1)
        ax_u.plot(etas, u42.mean(axis=-1), ls="--", color=col,
                  lw=0.9, alpha=0.85)
    ax_a.set_ylim(-0.03, 1.05)
    ax_a.set_xlabel(r"noise $\eta$"); ax_a.set_ylabel(r"$\langle\varphi\rangle$")
    ax_a.tick_params(labelsize=8)
    ax_a.set_title(r"(a) order parameter"
                   "\n"
                   r"$\alpha=1$ solid, $\alpha=2$ dashed",
                   fontsize=9)
    ax_a.legend(loc="upper right", fontsize=6, framealpha=0.9, ncol=2)
    ax_b.set_xlabel(r"noise $\eta$"); ax_b.set_ylabel(r"$\chi$")
    ax_b.tick_params(labelsize=8)
    ax_b.set_title(r"(b) susceptibility"
                   "\n"
                   r"$\alpha=1$ solid, $\alpha=2$ dashed",
                   fontsize=9)
    ax_u.axhline(2.0 / 3.0, ls=":", c="grey", lw=0.7)
    ax_u.set_xlabel(r"noise $\eta$"); ax_u.set_ylabel(r"$U_4$")
    ax_u.tick_params(labelsize=8)
    ax_u.set_title(r"(c) Binder cumulant"
                   "\n"
                   r"$\alpha=1$ solid, $\alpha=2$ dashed",
                   fontsize=9)

    # Panel (d): chi_max(L), bootstrap slope per alpha. Seed
    # 20260511 matches tools/update_chiffres.py so the slopes
    # printed here are byte-identical to the numbers substituted
    # into the manuscript.
    rng = np.random.default_rng(20260511)
    n_seeds = min(lev["chi"].shape[-1] for lev in levers)
    for ia, alabel in [(0, r"$\alpha=1$"), (1, r"$\alpha=2$")]:
        chi_max = np.zeros((n_L, n_seeds))
        for iL, lev in enumerate(levers):
            chi_max[iL] = lev["chi"][ia, :, :n_seeds].max(axis=0)
        s, se, ym, ys = _bootstrap_slope(Ls7, chi_max, rng=rng)
        col = A_COL[float(alphas[ia])]
        ax_c.errorbar(Ls7, ym, yerr=ys, fmt="o", color=col,
                      ms=5, capsize=3, lw=1.0,
                      label=fr"{alabel}: $L^{{{s:.2f}\pm{se:.2f}}}$")
        Lf = np.linspace(Ls7[0] * 0.9, Ls7[-1] * 1.1, 50)
        b = np.polyfit(np.log(Ls7), np.log(ym), 1)
        ax_c.plot(Lf, np.exp(b[1]) * Lf ** b[0], "--", color=col, lw=0.8)
    ax_c.set_xscale("log"); ax_c.set_yscale("log")
    ax_c.set_xticks(list(Ls7))
    ax_c.set_xticklabels([str(int(L)) for L in Ls7], fontsize=7)
    ax_c.minorticks_off()
    ax_c.set_xlabel(r"$L$")
    ax_c.set_ylabel(r"$\chi_{\max}$")
    ax_c.legend(loc="upper left", fontsize=7)
    ax_c.set_title(r"(d) FSS of $\chi_{\max}$", fontsize=9)
    ax_c.tick_params(labelsize=8)

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

    # v2 multi-L calibrated FSS, extended with L = 64, 91, 128 so
    # the calibrated lever matches the bare-eta 7-size lever of
    # Sec. III A. Each extension file carries its own L slice.
    z2 = np.load(DATA / "fss_calibrated_v2.npz")
    Ls = z2["Ls"]; alphas2 = z2["alphas"]; V2 = z2["V_targets"]
    eta_table2 = z2["eta_table"]
    chi2 = z2["chi"]   # (L, alpha, V, seed)
    n_seed_v2 = chi2.shape[-1]
    for L_ext in (64, 91, 128):
        p = DATA / f"fss_calibrated_L{L_ext}.npz"
        if p.exists():
            zE = np.load(p)
            chi_E = zE["chi"][..., :n_seed_v2]   # (alpha, V, seed)
            Ls = np.concatenate([Ls, [float(zE["L"])]])
            chi2 = np.concatenate([chi2, chi_E[None, ...]], axis=0)

    fig, axes = plt.subplots(1, 4,
                             figsize=(style.DOUBLE_COL[0] * 1.3375, 2.5))
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
        ax.tick_params(labelsize=8)
    ax_a.set_title("(a) calibration", fontsize=9)
    ax_phi.set_title("(b) order vs V (L=15)", fontsize=9)
    ax_chi.set_title("(c) susceptibility vs V (L=15)", fontsize=9)

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
    ax_fss.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax_fss.tick_params(labelsize=8)
    ax_fss.set_title("(d) FSS at fixed V", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig_calibrated.pdf")


if __name__ == "__main__":
    fig_fss()
    fig_calibrated()
