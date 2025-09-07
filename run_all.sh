#!/usr/bin/env bash
set -euo pipefail

# Create and use local venv if not active
if [ -z "${VIRTUAL_ENV:-}" ]; then
  python -m venv .venv
  source .venv/bin/activate
fi

pip install -U pip
pip install -r requirements.txt

# IMPORTANT: place your dataset CSV in the repo root (default expected by your script)
# e.g., Final_MMG_Dmax_dataset.csv

# Run training/calibration and then inverse design
python "scripts/Main_code.py"
# Some repositories use a second inverse-design script with a space in its name:
if [ -f "scripts/Main_code (1).py" ]; then
  python "scripts/Main_code (1).py"
fi

echo "All done."
