# Certified, ε‑Robust Inverse Design for Alloy Dmax

This repository packages **scripts** for robust, conformal, Bayesian optimization–driven inverse design of alloy **Dmax** (maximum castable diameter). The code trains quantile models on log‑Dmax, applies **conformal calibration** and **composition‑jitter ε certification**, and then optimizes a **conservative lower bound** via **Bayesian optimization** (ET and GP backends).

---

## Features (strictly matching the code)

- **Training & calibration** on log‑Dmax (quantile CatBoost).
- **Conformal subtraction** → certified lower bound \(L_{robust}\) in **mm**.
- **ε‑robustness** via Dirichlet jitter on the composition simplex (ℓ₁ at.% radius).
- **Inverse design** with **ET** and **GP** BO backends; two‑stage robustness (cheap → high‑fidelity rescoring).
- **Novelty & diversity** gating (at.% ℓ₁ to reference set, and batch‑diversity).
- **Exceedance reporting** vs dataset max (e.g., 35 mm) with **conservative CIs**.
- **ε‑sensitivity** curves and **jitter‑cloud** diagnostics (plot + CSV).
- **Baselines/ablations**: ET vs GP, prediction‑gate fixed vs adaptive.

---

## Repository layout

```
scripts/
  Main_code.py            # training + calibration and core utilities + inverse design

docs/
  FIGURE_MAP.md           # where each figure/CSV is written (from the scripts)
  REPRODUCIBILITY.md      # notes on environment, seeds, manifests

run_all.sh                # one-command runner (pip + both scripts)
Makefile                  # make setup | run | design | test
requirements.txt          # pip environment
environment.yml           # conda environment
LICENSE                   # MIT
README.md                 # this file
.editorconfig, .gitattributes, .gitignore
```

Outputs are written under `project_output/` (git‑ignored), mirroring the paths used in your scripts (e.g., `project_output/reports`, `project_output/source_data`, `project_output/data/designed`).

---

## Quick start

```bash
# create venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt

# Place your dataset CSV in the repo root if the script expects it
# (e.g., Final_MMG_Dmax_dataset.csv), or adjust the path in the script.

# Run full pipeline (training → inverse design)
bash run_all.sh
```

Targets run separately:

```bash
make run     # python "scripts/Main_code.py"
make design  # python "scripts/Main_code (1).py"
make test    # pytest -q
```

> CI only runs `pytest` to avoid data‑dependent failures on GitHub Actions.

---

## Data

Place your dataset CSV in the repository root (default expected by your scripts). If you cannot distribute the real data, provide a schema/stub (column names/types) so others can reproduce the pipeline structure.

---

## Key artifacts (non‑exhaustive)

From **Main_code.py** (training/calibration):
- `project_output/reports/fig_bo_pass_counts_vs_threshold.png`
- `project_output/reports/bo_multithreshold_summary_all.csv`
- `project_output/reports/bo_multithreshold_summary_novel.csv`
- `project_output/data/designed/advanced_bo_pool.csv`
- `project_output/data/designed/advanced_pour_list_all_ge_<D>mm.csv`

From inverse design:
- `project_output/source_data/bo_trace_forest.csv`, `bo_trace_gp.csv`
- `project_output/data/designed/bo_tried_all_<backend>.csv`
- `project_output/data/designed/bo_prefiltered_<backend>.csv`
- `project_output/data/designed/advanced_candidates_pred_ge_35mm_all_prethin.csv`
- `project_output/data/designed/advanced_candidates_pred_ge_35mm_all.csv`
- `project_output/data/designed/advanced_candidates_pred_ge_35mm_novel.csv`
- `project_output/data/designed/advanced_bo_pool_<backend>.csv`
- `project_output/reports/fig_bo_backend_bestL.png`
- `project_output/reports/pred_gate_ablation.csv`
- `project_output/reports/eps_sensitivity_top30.csv`
- `project_output/reports/eps_sensitivity_summary.csv`
- `project_output/reports/fig_eps_sensitivity.png`
- `project_output/reports/fig_jitter_cloud_top1.png`
- `project_output/reports/jitter_cloud_top1.csv`
- `project_output/reports/element_constraints_summary.csv`

See `docs/FIGURE_MAP.md` for a fuller map.

---

## Reproducibility

- **Environment:** `requirements.txt` (pip) or `environment.yml` (conda).
- **Seeds & CRN:** scripts define `SEED` and use common‑random‑numbers for robustness; manifests are written under `project_output/reports/`.
- **Determinism:** for bitwise reproducibility, pin BLAS threads (`OMP_NUM_THREADS=1`, etc.).
- **Unit tests:** `pytest -q` (monotonicity of \(L_{robust}\) vs ε; ε=0,K=1 consistency).

---

## License

MIT (see `LICENSE`).
