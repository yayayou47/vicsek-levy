"""
Regenerate fig_synthesis.pdf with panel (f) using L=64 chi_max
(via the per-seed and L=64 archives) and seed-bootstrap error bars.
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


def main():
    fig, axes = plt.subplots(2, 3, figsize=(style.DOUBLE_COL[0], 4.6))
    axes = axes.flatten()
    palette = ["#3aa040", "#d76f3a", "#1f4ea1"]
    rng = np.random.default_rng(0)

    def styled(ax, title):
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([1.0, 1.5, 2.0])

    def _bin_phi(eta_arr, phi_arr, eta_grid):
        return np.array([
            phi_arr[(eta_arr >= eta_grid[k])
                    & (eta_arr < eta_grid[k + 1])].mean()
            if ((eta_arr >= eta_grid[k])
                & (eta_arr < eta_grid[k + 1])).any()
            else np.nan for k in range(len(eta_grid) - 1)])

    # 1. Hysteresis area with block-bootstrap error
    ax = axes[0]
    z = np.load(DATA / "hysteresis.npz")
    alphas = z["alphas"]; eta_path = z["eta_path"]
    direction = z["direction"]; phi_traj = z["phi_traj"]
    eta_grid = np.linspace(eta_path.min(), eta_path.max(), 33)
    centers = 0.5 * (eta_grid[:-1] + eta_grid[1:])
    areas = []
    area_se = []
    block = 200
    n_boot = 400
    for ia in range(len(alphas)):
        up_idx = direction > 0; dn_idx = direction < 0
        up_eta = eta_path[up_idx]; dn_eta = eta_path[dn_idx]
        up_phi = phi_traj[ia, up_idx]; dn_phi = phi_traj[ia, dn_idx]
        u_bin = _bin_phi(up_eta, up_phi, eta_grid)
        d_bin = _bin_phi(dn_eta, dn_phi, eta_grid)
        ok = ~(np.isnan(u_bin) | np.isnan(d_bin))
        areas.append(float(np.trapezoid(np.abs(u_bin[ok] - d_bin[ok]),
                                         centers[ok])))
        # block bootstrap on the contiguous up/down ramp time series
        nu = len(up_eta); nd = len(dn_eta)
        boot_areas = np.empty(n_boot)
        n_blocks_u = max(1, nu // block)
        n_blocks_d = max(1, nd // block)
        for b in range(n_boot):
            iu = rng.integers(0, max(1, nu - block), size=n_blocks_u)
            ide = rng.integers(0, max(1, nd - block), size=n_blocks_d)
            ue = np.concatenate([up_eta[s:s + block] for s in iu])
            up_ = np.concatenate([up_phi[s:s + block] for s in iu])
            de = np.concatenate([dn_eta[s:s + block] for s in ide])
            dp_ = np.concatenate([dn_phi[s:s + block] for s in ide])
            ub = _bin_phi(ue, up_, eta_grid)
            db = _bin_phi(de, dp_, eta_grid)
            ok_b = ~(np.isnan(ub) | np.isnan(db))
            if ok_b.sum() > 3:
                boot_areas[b] = np.trapezoid(
                    np.abs(ub[ok_b] - db[ok_b]), centers[ok_b])
            else:
                boot_areas[b] = np.nan
        area_se.append(float(np.nanstd(boot_areas)))
    ax.errorbar(alphas, areas, yerr=area_se, fmt="o-",
                color=palette[0], ms=5, lw=1.2, capsize=3)
    ax.set_ylabel(r"hysteresis area")
    styled(ax, "(a) hysteresis area")

    # 2. Band index
    ax = axes[1]
    zb = np.load(DATA / "bands.npz")
    ax.plot(zb["alphas"], zb["band_idx"], "s-", color=palette[1],
            ms=5, lw=1.2)
    ax.axhline(1.0, ls=":", c="grey", lw=0.6)
    ax.set_ylabel(r"band index $b$")
    styled(ax, "(b) band amplitude")

    # 3. Correlation length xi
    ax = axes[2]
    zc = np.load(DATA / "correlations.npz")
    alpha_c = zc["alphas"]; eta_c = zc["etas"]; xi = zc["xi"]
    near_crit = {1.0: 0.05, 1.5: 0.10, 2.0: 0.15}
    xi_pts = []
    for ia, a in enumerate(alpha_c):
        ie = int(np.argmin(np.abs(eta_c - near_crit[float(a)])))
        xi_pts.append(xi[ia, ie])
    ax.plot(alpha_c, xi_pts, "D-", color=palette[2], ms=5, lw=1.2)
    ax.set_ylabel(r"$\xi$ at $\eta_c(\alpha)$")
    styled(ax, "(c) correlation length")

    # 4. Spatial-MSD exponent with fit-window jackknife error
    ax = axes[3]
    zd = np.load(DATA / "diffusion.npz")
    a_d = zd["alphas"]; eta_d = zd["etas"]
    t = zd["t"].astype(float); msd_x = zd["msd_x"]
    ie = int(np.argmin(np.abs(eta_d - 0.15)))
    fit = (t >= 50) & (t <= 1500)

    def _slope_jk(t_, y_):
        idx = np.where(fit)[0]
        slopes = []
        for k in range(len(idx)):
            keep = np.delete(idx, k)
            s, _ = np.polyfit(np.log(t_[keep]), np.log(y_[keep]), 1)
            slopes.append(s)
        slopes = np.asarray(slopes)
        n = len(slopes)
        s_mean = slopes.mean()
        s_se = np.sqrt((n - 1) / n * np.sum((slopes - s_mean) ** 2))
        return float(s_mean), float(s_se)

    gam_x = []; gam_se = []
    for ia in range(len(a_d)):
        m, se = _slope_jk(t, msd_x[ia, ie])
        gam_x.append(m); gam_se.append(se)
    ax.errorbar(a_d, gam_x, yerr=gam_se, fmt="v-",
                color=palette[0], ms=5, lw=1.2, capsize=3)
    ax.axhline(1.0, ls=":", c="grey", lw=0.6, label="diffusive")
    ax.axhline(2.0, ls="--", c="grey", lw=0.6, label="ballistic")
    ax.set_ylabel(r"$\gamma_x$ at $\eta = 0.15$")
    ax.legend(fontsize=6, loc="lower right")
    styled(ax, "(d) spatial MSD exponent")

    # 5. Angular diffusivity with bootstrap error on the fit-window mean
    ax = axes[4]
    msd_th = zd["msd_theta"]
    D_th = []; D_se = []
    n_b = 400
    idx_fit = np.where(fit)[0]
    for ia in range(len(a_d)):
        y = msd_th[ia, ie][fit]
        D_th.append(float(y.mean() / t[fit].mean()))
        boot = np.empty(n_b)
        for b in range(n_b):
            sample = rng.choice(len(y), size=len(y), replace=True)
            boot[b] = y[sample].mean() / t[idx_fit[sample]].mean()
        D_se.append(float(boot.std()))
    ax.errorbar(a_d, D_th, yerr=D_se, fmt="^-",
                color=palette[1], ms=5, lw=1.2, capsize=3)
    ax.set_ylabel(r"$D_\theta$ at $\eta = 0.15$")
    styled(ax, "(e) angular diffusivity")

    # 6. chi_max at L = 64 (5-seed bootstrap)
    ax = axes[5]
    z64 = np.load(DATA / "fss_L64.npz")
    a_f = z64["alphas"]
    chi_64_per_seed = z64["chi"].max(axis=1)  # (alpha, seed)
    chi_mean = chi_64_per_seed.mean(axis=-1)
    chi_se = chi_64_per_seed.std(axis=-1) / np.sqrt(chi_64_per_seed.shape[-1])
    ax.errorbar(a_f, chi_mean, yerr=chi_se, fmt="*-",
                color=palette[2], ms=7, lw=1.2, capsize=3)
    ax.set_ylabel(r"$\chi_{\max}$ at $L = 64$")
    styled(ax, "(f) FSS susceptibility peak")

    fig.tight_layout()
    _save(fig, "fig_synthesis.pdf")


if __name__ == "__main__":
    main()
