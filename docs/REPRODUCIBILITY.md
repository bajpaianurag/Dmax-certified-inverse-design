# Reproducibility Notes

1. **Environment**
   - Use `requirements.txt` (pip) or `environment.yml` (conda).

2. **Determinism**
   - Scripts define `SEED` and use controlled RNGs (including CRN for robustness objective).
   - Ensure single-threaded execution if you need bitwise determinism (set MKL/OMP env vars).

3. **Data**
   - Place your dataset CSV in the repo root (default expected by scripts).

4. **Manifests**
   - Scripts write per-run manifests (`reports/run_manifest_*.json`) and BO traces (`source_data/bo_trace_*.csv`).

5. **Tests**
   - `tests/test_cert_objective.py` checks key math properties without modifying your code paths.
