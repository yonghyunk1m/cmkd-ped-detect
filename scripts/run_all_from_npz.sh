#!/bin/bash
# Everything that the four prediction dumps unlock. No GPU, no dataset needed.
set -e
cd "$(dirname "$0")/.."
echo "### Table 1 rows, significance, calibration ###"
python scripts/verify_paper_numbers.py
echo; echo "### Per-fold ECE / Brier + paired tests ###"
python scripts/ece_significance.py
echo; echo "### Threshold ablation + the calibration defense ###"
python scripts/threshold_from_npz.py
