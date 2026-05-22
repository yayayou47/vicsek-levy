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
| `vicsek.py`        | Zonal Vicsek–Couzin with α-stable angular noise and full $360^\circ$ vision. Repulsion zone $\mathcal R_i$ ($d<R_r$), alignment annulus $\mathcal A_i$ ($R_r\le d<R_a$). Numba-accelerated cell-list neighbours. |
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

`fss_runner.py` is a shared FSS helper (not a driver): it holds the
`FSSConfig` dataclass and the parallel `run_fss_sweep` machinery that the
thin per-$L$ FSS wrappers below import.

| Driver | Output | Purpose |
|---|---|---|
| **Phase diagrams**            |                            |                                                                                       |
| `run_3d_sweep.py`             | `sweep_3d.npz`             | $\varphi(\eta, R_r, R_a)$ at three $\alpha$.                                          |
| `run_phasecurve.py`           | `phase_curve.npz`          | Smooth $\eta_c(\alpha)$ on six $\alpha$, three seeds.                                  |
| `run_gnf.py`                  | `gnf.npz`                  | Giant number fluctuations, $L=30$ (panel (d) of the phase-curve figure).              |
| **FSS — metric model**        |                            |                                                                                       |
| `run_fss_smallL.py`           | `fss_L{15,22,30,45,64}.warm20000.npz` | Per-seed FSS at the five small sizes with the long warm-up.               |
| `run_fss_L91.py`              | `fss_L91.npz`              | Intermediate FSS point at $L=91$ (thin wrapper around `fss_runner`).                   |
| `run_fss_L128.py`             | `fss_L128.npz`             | FSS endpoint at $L=128$ (thin wrapper around `fss_runner`).                            |
| `run_fss_L64.py`              | `fss_L64.npz`              | Pilot $L=64$ run, refined $\eta$ grid; feeds the synthesis figure.                    |
| **Calibration**               |                            |                                                                                       |
| `run_calibrated.py`           | `calibrated_sweep.npz`     | $\varphi(V), \chi(V)$ at $L=15$ for the wrapped circular variance $V$.                |
| `run_calibrated_fss.py`       | `fss_calibrated_v2.npz`    | Multi-$L$ FSS at fixed $V$ (controls for amplitude effects).                          |
| `run_calibrated_anyL.py`      | `fss_calibrated_L{L}.npz`  | $V$-calibrated FSS at a single user-specified $L$ ($L\in\{91,128\}$ extensions).       |
| `run_calibrated_L64.py`       | `fss_calibrated_L64.npz`   | $V$-calibrated FSS at $L=64$, extending the calibrated lever.                          |
| **Bulk-fluid signatures**     |                            |                                                                                       |
| `run_correlations.py`         | `correlations.npz`         | Velocity correlation $C_v(r)$ and pair correlation $g(r)$.                            |
| `run_clusters.py`             | `clusters.npz`             | Connected-component cluster-size distribution (feeds the single-particle figure).      |
| `run_bands.py`                | `bands.npz`                | Travelling-band detection (metric).                                                   |
| `run_bands_topo.py`           | `bands_topo.npz`           | Travelling-band detection (topological — confirms band-free).                         |
| `run_diffusion.py`            | `diffusion.npz`            | Single-particle angular and spatial MSD.                                              |
| **Hysteresis**                |                            |                                                                                       |
| `run_hysteresis.py`           | `hysteresis.npz`           | $\eta$-ramp loop, $T_{up}=T_{dn}=8000$.                                              |
| `run_hysteresis_slow.py`      | `hysteresis_slow.npz`      | Slower ramp ($T=32000$) for ramp-speed control.                                       |
| **Order-parameter PDFs**      |                            |                                                                                       |
| `run_orderpdf.py`             | `orderpdf.npz`             | $P(\langle\varphi\rangle)$ at four corners (metric/topological × $\alpha\in\{1,2\}$). |
| `run_orderpdf_k.py`           | `orderpdf_k.npz`           | $P(\langle\varphi\rangle)$ for $k\in\{4,6,10\}$ (topological).                         |
| `run_order_snapshots.py`      | `order_snapshots.npz`      | Particle snapshots for the 2×3 ordered / near-critical / disordered grid.              |
| **Topological alignment**     |                            |                                                                                       |
| `run_topological.py`          | `topo_fss.npz`             | Mini-FSS, 4 sizes × 2 $\alpha$ × 5 seeds.                                              |
| `run_topo_a15_smallL.py`      | `topo_fss_a15.npz`         | $\alpha=1.5$ slice that harmonises `topo_fss.npz` to three $\alpha$.                   |
| `run_topo_anyL.py`            | `topo_L{L}_a3.npz`         | Topological FSS at a single $L$ ($L\in\{64,91,128\}$ extensions, 3 $\alpha$).          |
| `run_topo_L64.py`             | `topo_L64.npz`             | Topological FSS at $L=64$ (legacy 2-$\alpha$ fallback for the extension).              |
| `run_topo_k.py`               | `topo_k_scan.npz`          | $\chi_{\max}(L)$ versus $k$ at $\alpha=1$.                                            |
| **Robustness**                |                            |                                                                                       |
| `run_robustness.py`           | `robustness.npz`           | $\varphi(\eta)$ under $v_0,\sigma$ variations × 5 seeds.                              |
| **Adaptive noise**            |                            |                                                                                       |
| `run_adaptive_perseed.py`     | `adaptive_perseed.npz`     | Density-adaptive $\alpha_i(\rho_{\rm local})$ FSS pilot, 5 seeds.                      |
| `run_adaptive_extension.py`   | `adaptive_perseed_ext.npz` | Extends the adaptive pilot to $L\in\{64,91,128\}$ for the 7-size lever.                |

---

## Figure builders (`src/`) → `figures/`

Four builder modules, each importing the simulators and reading from
`data/`. Run order does not matter: every builder is independent.

```bash
cd src/
python make_figures.py        # batch builder for the main + SI figures
python make_figures_v2.py     # rebuilds fig_fss.pdf and fig_calibrated.pdf
python make_figures_v3.py     # fig_hysteresis, fig_adaptive_pilot, fig_order_snapshots
python make_synthesis_v2.py   # rebuilds fig_synthesis.pdf
```

`make_figures.py` exposes one function per figure (e.g. `fig_phase_curve`,
`fig_topological`, `fig_robustness`); see its `main()` for the canonical
calling sequence.

The repository ships exactly the 23 figures of the submitted manuscript and
its Supplemental Material. Since the original release the cluster figure was
merged into the single-particle diffusion figure and the giant-number-
fluctuations figure was merged into the phase-curve figure, so `clusters.npz`
and `gnf.npz` are still produced and read as data even though no standalone
`fig_clusters` / `fig_gnf` exist any more.

| Figure (PDF + PNG)            | Builder & function                        | Data archives                                            |
|---|---|---|
| **Main text**                 |                                           |                                                          |
| `fig_model_schematic`         | `make_figures.fig_model_schematic`        | (drawn programmatically)                                 |
| `fig_noise_pdf`               | `make_figures.fig_noise_pdf`              | (drawn from `noise.py`)                                  |
| `fig_rule_inertia`            | `make_figures.fig_rule_inertia`           | (drawn programmatically)                                 |
| `fig_rule_repulsion`          | `make_figures.fig_rule_repulsion`         | (drawn programmatically)                                 |
| `fig_rule_alignment`          | `make_figures.fig_rule_alignment`         | (drawn programmatically)                                 |
| `fig_order_snapshots`         | `make_figures_v3.fig_order_snapshots`     | `order_snapshots.npz`                                    |
| `fig_fss`                     | `make_figures_v2.fig_fss`                 | `fss_L{15,22,30,45,64}.warm20000.npz`, `fss_L91.npz`, `fss_L128.npz` |
| `fig_hysteresis`              | `make_figures_v3.fig_hysteresis`          | `hysteresis.npz`, `hysteresis_slow.npz`                  |
| `fig_calibrated`              | `make_figures_v2.fig_calibrated`          | `calibrated_sweep.npz`, `fss_calibrated_v2.npz`, `fss_calibrated_L{64,91,128}.npz` |
| `fig_phase_curve`             | `make_figures.fig_phase_curve`            | `phase_curve.npz`, `gnf.npz`                             |
| `fig_bands`                   | `make_figures.fig_bands`                  | `bands.npz`                                              |
| `fig_correlations`            | `make_figures.fig_correlations`           | `correlations.npz`                                       |
| `fig_diffusion`               | `make_figures.fig_diffusion`              | `diffusion.npz`, `clusters.npz`                          |
| `fig_topological`             | `make_figures.fig_topological`            | `topo_fss.npz`, `topo_fss_a15.npz`, `topo_L{64,91,128}_a3.npz`, `topo_L64.npz` |
| `fig_orderpdf`                | `make_figures.fig_orderpdf`               | `orderpdf.npz`                                           |
| **Supplemental Material**     |                                           |                                                          |
| `fig_3d_phase`                | `make_figures.fig_3d_phase`               | `sweep_3d.npz`                                           |
| `fig_snapshots`               | `make_figures.fig_snapshots`              | (live simulation, no npz)                                |
| `fig_bands_topo`              | `make_figures.fig_bands_topo`             | `bands_topo.npz`                                         |
| `fig_synthesis`               | `make_synthesis_v2.main`                  | `hysteresis.npz`, `bands.npz`, `diffusion.npz`, `fss_L64.npz` |
| `fig_topo_k`                  | `make_figures.fig_topo_k`                 | `topo_k_scan.npz`                                        |
| `fig_orderpdf_k`              | `make_figures.fig_orderpdf_k`             | `orderpdf_k.npz`, `orderpdf.npz`                         |
| `fig_adaptive_pilot`          | `make_figures_v3.fig_adaptive_pilot`      | `adaptive_perseed.npz`, `adaptive_perseed_ext.npz`       |
| `fig_robustness`              | `make_figures.fig_robustness`             | `robustness.npz`                                         |

---

## Data formats

Every `.npz` archive is a flat `numpy.lib.npyio.NpzFile`. Inspect any of
them with

```python
import numpy as np
z = np.load("data/fss_L30.warm20000.npz")
print({k: z[k].shape for k in z.files})
```

Per-seed archives use a trailing seed axis: e.g. `phi.shape ==
(n_alpha, n_eta, n_seed)` in the `fss_L*.warm20000.npz` files, and similarly
for `topo_fss.npz`, `robustness.npz`, `adaptive_perseed.npz`, `fss_L64.npz`,
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
python run_3d_sweep.py
python run_phasecurve.py
python run_gnf.py
# FSS metric (≈10 h, dominated by the large sizes)
python run_fss_smallL.py
python run_fss_L91.py
python run_fss_L128.py
python run_fss_L64.py
# Calibration (≈1 h)
python run_calibrated.py
python run_calibrated_fss.py
python run_calibrated_L64.py
python run_calibrated_anyL.py    # repeat for L = 91, 128
# Bulk fluid (≈1 h)
python run_correlations.py
python run_clusters.py
python run_bands.py
python run_bands_topo.py
python run_diffusion.py
# Hysteresis (≈1 h)
python run_hysteresis.py
python run_hysteresis_slow.py
# Order-parameter PDFs and snapshots (≈40 min)
python run_orderpdf.py
python run_orderpdf_k.py
python run_order_snapshots.py
# Topological FSS (≈1 h on 8 cores)
python run_topological.py
python run_topo_a15_smallL.py
python run_topo_L64.py
python run_topo_anyL.py          # repeat for L = 64, 91, 128 at 3 alphas
python run_topo_k.py
# Robustness (≈10 min on 8 cores)
python run_robustness.py
# Adaptive (≈1 h)
python run_adaptive_perseed.py
python run_adaptive_extension.py
# Then the figures:
python make_figures.py        # batch builder for the main + SI figures
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
- Vision: full $360^\circ$ (no blind sector).
- Repulsion / alignment radii: $R_r = 0.45$, $R_a = 0.7$ (zonal
  model); $k = 6$ nearest neighbours in the topological model.
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
