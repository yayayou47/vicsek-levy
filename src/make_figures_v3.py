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
DATA = HERE.parent / "data"
FIG = HERE.parent / "figures"
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

    # 4 panels: 3 alphas at fast ramp + 1 panel comparing fast vs slow
    fig, axes = plt.subplots(1, 4,
                             figsize=(style.DOUBLE_COL[0], 2.4),
                             sharey=True)
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
        area = float(np.trapezoid(pi_up - pi_dn, eg))
        gap = float(np.max(pi_up - pi_dn))
        ax.set_title(fr"$\alpha = {alphas_fast[ia]:.1f}$"
                     "\n"
                     fr"area $= {area:.3f}$, gap $= {gap:.2f}$",
                     fontsize=8)
        ax.set_xlabel(r"noise $\eta$")
        if ia == 0:
            ax.set_ylabel(r"$\langle\varphi\rangle$")
            ax.legend(loc="upper right", fontsize=7)
        ax.set_ylim(-0.02, 1.05); ax.tick_params(labelsize=7)

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
    ax.set_title(r"slow ramp $T = 32000$" "\n"
                 r"area: $\alpha{=}2{:}\,0.006$, $\alpha{=}1{:}\,0$",
                 fontsize=8)
    ax.set_xlabel(r"noise $\eta$")
    ax.set_ylim(-0.02, 1.05); ax.tick_params(labelsize=7)
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

    fig = plt.figure(figsize=(style.DOUBLE_COL[0], 5.0))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.32,
                          height_ratios=[1.2, 1.0])
    cmap = plt.get_cmap("plasma")
    cols = [cmap(0.15 + 0.7 * i / 3) for i in range(4)]

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
            ax.fill_between(etas, m - s, m + s, color=cols[iL],
                            alpha=0.18, lw=0)
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
        ax.set_xlabel(r"noise $\eta$"); ax.tick_params(labelsize=7)
    ax_p.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.9)

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
    ax_a.tick_params(labelsize=7); ax_as.tick_params(labelsize=7)

    # FSS log-log fit panel: chi_max(L) per mode with bootstrap
    rng = np.random.default_rng(0)
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
    ax_fss.tick_params(labelsize=7)

    fig.tight_layout()
    _save(fig, "fig_adaptive_pilot.pdf")


if __name__ == "__main__":
    fig_hysteresis()
    fig_adaptive_pilot()
