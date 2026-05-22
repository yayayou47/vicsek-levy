# vicsek-levy

[![tests](https://github.com/yayayou47/vicsek-levy/actions/workflows/test.yml/badge.svg)](https://github.com/yayayou47/vicsek-levy/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Numerical experiments and figure pipeline for the manuscript

> **Interaction topology controls whether heavy-tailed noise reshapes the Vicsek flocking transition**
> Yaya Youssouf Yaya & Djibrine Abakar (submitted to *Physical Review E*, 2026)

The Gaussian angular kick of the canonical Vicsek model is replaced by a
symmetric α-stable (Lévy) law of tail index $\alpha\in(0,2]$. This repository
contains the simulation code (NumPy + Numba), the run drivers that produce
every numerical result quoted in the paper, the resulting `.npz` archives,
the figure builders, and the camera-ready figures.

The repository is self-contained: from a fresh clone, every figure can be
rebuilt either from the deposited `.npz` archives (cheap; minutes) or from
scratch by re-running the simulation drivers (expensive; hours).

---

## Repository layout

```
vicsek-levy/
├── README.md                this file
├── LICENSE                  MIT
├── requirements.txt         pinned minima
├── .gitignore
├── src/                     simulation core, run drivers, figure builders
├── tests/                   pytest suite for noise + dynamics
├── data/                    .npz archives (one per run driver)
└── figures/                 PDF + PNG of every paper figure
```

`src/` is intentionally flat so that every script can import the simulators
with a plain `from vicsek import …` / `from topological import …`.

---

## Installation

```bash
git clone https://github.com/<user>/vicsek-levy.git
cd vicsek-levy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11. The simulators rely on Numba JIT for the inner
neighbour loop; the first call to `Vicsek.step()` triggers compilation
(≈10 s warm-up).

Run the test suite:

```bash
pytest tests/
```

---

## The simulators (`src/`)

| File | Role |
|---|---|
| `vicsek.py`        | Zonal Vicsek–Couzin with α-stable angular noise. Repulsion zone $\mathcal R_i$ ($d<R_r$), alignment annulus $\mathcal A_i$ ($R_r\le d<R_a$), optional blind sector $\beta$. Numba-accelerated cell-list neighbours. |
| `topological.py`   | Same dynamics but alignment over the $k$ topologically nearest visible neighbours via a `cKDTree`. |
| `noise.py`         | Symmetric α-stable random variates (Chambers–Mallows–Stuck) and the wrapped circular variance $V$ used for noise calibration. |
| `analysis.py`      | Per-snapshot observables: polarisation $\varphi$, cluster decomposition, density grid. |
| `style.py`         | Shared matplotlib style (PRE/PRL column widths, cream-on-charcoal palette). |

---

## Run drivers (`src/run_*.py`) → data (`data/*.npz`)

Every driver writes a single `.npz` archive into `data/`. All drivers are
self-contained and parameter-explicit at the top of `main()`. Run as

```bash
cd src/
python run_<driver>.py
```

| Driver | Output | Purpose |
|---|---|---|
| **Phase diagrams**            |                            |                                                                                       |
| `run_phase.py`                | `phase_sweep.npz`          | $\varphi(\eta,\alpha)$ time series at $L=15$.                                          |
| `run_3d_sweep.py`             | `sweep_3d.npz`             | $\varphi(\eta, R_r, R_a)$ at three $\alpha$.                                          |
| `run_phasecurve.py`           | `phase_curve.npz`          | Smooth $\eta_c(\alpha)$ on six $\alpha$, three seeds.                                  |
| **FSS — metric model**        |                            |                                                                                       |
| `run_fss_sweep.py`            | `fss_sweep.npz`            | First-pass FSS, four sizes, mean only (legacy).                                       |
| `run_fss_perseed.py`          | `fss_perseed.npz`          | Per-seed FSS (5 seeds × 4 sizes) — main FSS source.                                   |
| `run_fss_L64.py`              | `fss_L64.npz`              | Extension to $L=64$ (5 seeds), refined $\eta$ grid.                                   |
| `run_fss_standard_vicsek.py`  | `fss_standard.npz`         | Sanity check with the canonical Vicsek update ($R_r=0$, no blind sector).             |
| **Calibration**               |                            |                                                                                       |
| `run_calibrated.py`           | `calibrated_sweep.npz`     | $\varphi(V), \chi(V)$ at $L=15$ for the wrapped circular variance $V$.                |
| `run_calibrated_fss.py`       | `fss_calibrated.npz`,<br>`fss_calibrated_v2.npz` | Multi-$L$ FSS at fixed $V$ (controls for amplitude effects).                          |
| **Bulk-fluid signatures**     |                            |                                                                                       |
| `run_gnf.py`                  | `gnf.npz`                  | Giant number fluctuations, $L=30$.                                                    |
| `run_gnf_L45.py`              | `gnf_L45.npz`              | Same observable at $L=45$ (size-dependence check).                                    |
| `run_correlations.py`         | `correlations.npz`         | Velocity correlation $C_v(r)$ and pair correlation $g(r)$.                            |
| `run_clusters.py`             | `clusters.npz`             | Connected-component cluster-size distribution $P(s)$.                                 |
| `run_bands.py`                | `bands.npz`                | Travelling-band detection (metric).                                                   |
| `run_bands_topo.py`           | `bands_topo.npz`           | Travelling-band detection (topological — confirms band-free).                         |
| `run_diffusion.py`            | `diffusion.npz`            | Single-particle angular and spatial MSD.                                              |
| **Hysteresis**                |                            |                                                                                       |
| `run_hysteresis.py`           | `hysteresis.npz`           | Slow $\eta$-ramp loop, $T_{up}=T_{dn}=8000$.                                          |
| `run_hysteresis_slow.py`      | `hysteresis_slow.npz`      | Slower ramp ($T=32000$) for ramp-speed control.                                       |
| **Order-parameter PDFs**      |                            |                                                                                       |
| `run_orderpdf.py`             | `orderpdf.npz`             | $P(\langle\varphi\rangle)$ at four corners (metric/topological × $\alpha\in\{1,2\}$). |
| `run_orderpdf_k.py`           | `orderpdf_k.npz`           | $P(\langle\varphi\rangle)$ for $k\in\{4,6,10\}$ (topological).                         |
| **Topological alignment**     |                            |                                                                                       |
| `run_topological.py`          | `topo_fss.npz`             | Mini-FSS, 4 sizes × 2 α × 5 seeds.                                                    |
| `run_topo_k.py`               | `topo_k_scan.npz`          | $\chi_{\max}(L)$ versus $k$ at $\alpha=1$.                                            |
| **Robustness**                |                            |                                                                                       |
| `run_robustness.py`           | `robustness.npz`           | $\varphi(\eta)$ under $v_0,\beta,\sigma$ variations × 5 seeds.                         |
| **Adaptive noise**            |                            |                                                                                       |
| `run_adaptive_perseed.py`     | `adaptive_perseed.npz`     | Density-adaptive $\alpha_i(\rho_{\rm local})$ FSS pilot, 5 seeds.                      |
| **Movies (not figures)**      |                            |                                                                                       |
| `run_movie.py`                | mp4                        | Single-regime movie.                                                                  |
| `run_movie_composite.py`      | mp4                        | Four-regime composite.                                                                |
| `manim_exposition.py`         | mp4                        | Manim figure-centric exposition (companion video).                                    |

The `*.bak` files in `data/` are pre-rerun snapshots and are excluded from
the repository (`.gitignore`).

---

## Figure builders (`src/`) → `figures/`

Three builder modules, each importing the simulators and reading from
`data/`. Run order does not matter: every builder is independent.

```bash
cd src/
python -c "from pathlib import Path; import make_figures as m; m.fig_gnf(Path('../data/gnf.npz'))"
python make_figures_v2.py     # rebuilds fig_fss.pdf and fig_calibrated.pdf
python make_figures_v3.py     # rebuilds fig_hysteresis.pdf and fig_adaptive_pilot.pdf
python make_synthesis_v2.py   # rebuilds fig_synthesis.pdf
```

`make_figures.py` exposes one function per figure (e.g. `fig_gnf`,
`fig_topological`, `fig_robustness`); see its `main()` for the canonical
calling sequence.

| Figure (PDF + PNG)            | Builder & function                        | Data archives                                            |
|---|---|---|
| `fig_noise_pdf`               | `make_figures.fig_noise_pdf`              | (drawn from `noise.py`)                                  |
| `fig_model_schematic`         | `make_figures.fig_model_schematic`        | (drawn programmatically)                                 |
| `fig_phase_curve`             | `make_figures.fig_phase_curve`            | `phase_curve.npz`                                        |
| `fig_3d_phase`                | `make_figures.fig_3d_phase`               | `sweep_3d.npz`                                           |
| `fig_fss`                     | `make_figures_v2.fig_fss_alpha2`          | `fss_perseed.npz`, `fss_L64.npz`                         |
| `fig_calibrated`              | `make_figures_v2.fig_calibrated`          | `calibrated_sweep.npz`, `fss_calibrated_v2.npz`          |
| `fig_gnf`                     | `make_figures.fig_gnf`                    | `gnf.npz`                                                |
| `fig_correlations`            | `make_figures.fig_correlations`           | `correlations.npz`                                       |
| `fig_clusters`                | `make_figures.fig_clusters`               | `clusters.npz`                                           |
| `fig_bands`                   | `make_figures.fig_bands`                  | `bands.npz`                                              |
| `fig_bands_topo`              | `make_figures.fig_bands_topo`             | `bands_topo.npz`                                         |
| `fig_diffusion`               | `make_figures.fig_diffusion`              | `diffusion.npz`                                          |
| `fig_hysteresis`              | `make_figures_v3.fig_hysteresis`          | `hysteresis.npz`, `hysteresis_slow.npz`                  |
| `fig_orderpdf`                | `make_figures.fig_orderpdf`               | `orderpdf.npz`                                           |
| `fig_orderpdf_k`              | `make_figures.fig_orderpdf_k`             | `orderpdf_k.npz`, `orderpdf.npz`                         |
| `fig_topological`             | `make_figures.fig_topological`            | `topo_fss.npz`                                           |
| `fig_topo_k`                  | `make_figures.fig_topo_k`                 | `topo_k_scan.npz`                                        |
| `fig_robustness`              | `make_figures.fig_robustness`             | `robustness.npz`                                         |
| `fig_adaptive_pilot`          | `make_figures_v3.fig_adaptive_pilot`      | `adaptive_perseed.npz`                                   |
| `fig_synthesis`               | `make_synthesis_v2.main`                  | (multi-archive aggregate)                                |
| `fig_snapshots`               | `make_figures.fig_snapshots`              | (live simulation, no npz)                                |

---

## Data formats

Every `.npz` archive is a flat `numpy.lib.npyio.NpzFile`. Inspect any of
them with

```python
import numpy as np
z = np.load("data/fss_perseed.npz")
print({k: z[k].shape for k in z.files})
```

Per-seed archives use a trailing seed axis: e.g. `phi.shape ==
(n_alpha, n_L, n_eta, n_seed)` in `fss_perseed.npz`, and similarly for
`topo_fss.npz`, `robustness.npz`, `adaptive_perseed.npz`, `fss_L64.npz`,
`fss_calibrated_v2.npz`. Older archives without a seed axis are kept for
backwards-compatible figure builders, which now detect both shapes.

The five fixed seeds used throughout are `{11, 23, 41, 67, 89}`.

---

## Reproducing every figure from scratch

The full pipeline (assuming 8 cores) takes roughly **8–12 hours** of wall
time. Most of it is the topological FSS and the slow hysteresis ramp.

```bash
cd src/
# Phase diagrams (≈30 min total)
python run_phase.py
python run_3d_sweep.py
python run_phasecurve.py
# FSS metric (≈2 h)
python run_fss_perseed.py
python run_fss_L64.py
python run_fss_standard_vicsek.py
# Calibration (≈1 h)
python run_calibrated.py
python run_calibrated_fss.py
# Bulk fluid (≈1 h)
python run_gnf.py
python run_gnf_L45.py
python run_correlations.py
python run_clusters.py
python run_bands.py
python run_bands_topo.py
python run_diffusion.py
# Hysteresis (≈1 h)
python run_hysteresis.py
python run_hysteresis_slow.py
# Order-parameter PDFs (≈30 min)
python run_orderpdf.py
python run_orderpdf_k.py
# Topological FSS (≈30 min on 8 cores)
python run_topological.py
python run_topo_k.py
# Robustness (≈10 min on 8 cores)
python run_robustness.py
# Adaptive (≈30 min)
python run_adaptive_perseed.py
# Then the figures:
python make_figures.py        # batch builder for the legacy figures
python make_figures_v2.py
python make_figures_v3.py
python make_synthesis_v2.py
```

Several drivers parallelise across seeds via `concurrent.futures.ProcessPoolExecutor`
with `n_workers = 8`; reduce the constant in the script if you have fewer
cores.

---

## Numerical conventions

- Box: square with periodic boundaries, side $L$, fixed density
  $\sigma = N/L^2 \simeq 2.22$.
- Speed: $v_0 = 0.05$ throughout, except in the $v_0$ axis of robustness.
- Repulsion / alignment: $R_r = 0.5$, $R_a = 0.7$ (zonal model);
  $k = 6$ in the topological model.
- Blind sector: $\beta = 30^\circ$ by default.
- Time steps: $n_{\rm warm} \in [600, 1500]$ followed by
  $n_{\rm meas} \in [400, 1500]$. Each driver pins its own values at the
  top of `main()`.
- Seeds: `{11, 23, 41, 67, 89}` (per-seed archives) or 3-seed subsets.
- Errors: every fitted slope quoted in the paper is bootstrap or jackknife
  on seeds; figure-level uncertainty is mean ± SEM ribbons.

---

## Citation

If you use the code or data, please cite

```bibtex
@article{YayaAbakar2026,
  author  = {Yaya Youssouf Yaya and Djibrine Abakar},
  title   = {Interaction topology controls whether heavy-tailed
             noise reshapes the Vicsek flocking transition},
  journal = {Physical Review E},
  year    = {2026},
  note    = {Submitted}
}
```

---

## Authors

- **Yaya Youssouf Yaya** — ENS N'Djaména / Université Assane Seck, Ziguinchor — `yy.y@zig.univ.sn` — [ORCID 0000-0003-0781-4923](https://orcid.org/0000-0003-0781-4923)
- **Djibrine Abakar** — Université Polytechnique de Mongo — `djibrine@upm.edu.td` — [ORCID 0000-0002-3904-1497](https://orcid.org/0000-0002-3904-1497)

## Acknowledgments

We thank the Collective Behaviour group at Uppsala University, led by
David J. T. Sumpter, and in particular Maksym Romenskyy, for inspiring
discussions during a research visit to their lab.

## License

MIT — see `LICENSE`.
