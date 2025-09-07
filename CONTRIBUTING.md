# Contributing

## Reproducibility first
- Do not change `scripts/Main_code.py` or `scripts/Main_code (1).py` unless you also update the README and add tests.
- Prefer adding new functionality in separate modules while preserving the original scripts.

## How to run locally
```bash
bash run_all.sh
```

## Tests
Run unit tests:
```bash
pytest -q
```

## Data
Place your dataset CSV at the repo root (e.g., `Final_MMG_Dmax_dataset.csv`) or adjust paths in the scripts.
