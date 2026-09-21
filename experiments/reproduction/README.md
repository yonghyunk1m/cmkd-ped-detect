# Reproducing the paper's numbers from the prediction dumps

These scripts reproduce the central results of the paper directly from the
committed per-session prediction dumps, so no retraining or GPU is required.
The four LOSO prediction dumps
(`results/predictions_loso/*.npz`, one per config: Baseline, LogitKD,
LogitKD_TFD, Hybrid_TFD) are committed via Git LFS and are the authoritative
source; each holds `session_<id>_probs` and `session_<id>_labels` for the five
held-out sessions.

## Files
- `recompute_from_npz.py` / `recompute_from_npz_results.txt` — from the npz
  dumps, recomputes per-fold Macro / No-Ped / Ped / F1 / PR-AUC / ECE / Brier
  reliability for the four configs and runs paired exact-permutation tests,
  percentile bootstrap CIs, and Holm correction against the baseline.
  Reproduces Table 1 (72.9 / 73.6 / 73.7 / 73.6), the positive-class ECE
  0.366 → 0.316, Brier reliability 0.163 → 0.110, and the predicted-Ped ratio
  0.404 → 0.33.
- `stats_calibration.py` / `stats_calibration_results.txt` — small-sample
  statistical strengthening of the calibration claims from
  `loso_per_fold_calibration.csv` (exact 2^5 sign-flip permutation p, paired
  Cohen's d with population std, folds-improved, bootstrap CI, Holm across the
  family; resolution is a control that should not move).
- `loso_per_fold_from_npz.csv` — the reproduced per-fold values (4 configs × 5 folds).
- `loso_per_fold_calibration.csv` — per-fold ECE / Brier reliability / resolution.
- `loso_per_fold_full.csv` — per-fold macro for the wider config set where it is
  available (incomplete for some configs; the npz dumps remain authoritative).

## Run
```bash
# needs numpy, scipy, scikit-learn
python experiments/reproduction/recompute_from_npz.py
python experiments/reproduction/stats_calibration.py
```
`recompute_from_npz.py` reads `results/predictions_loso/` by default; set
`NPZ_DIR` to override.

## Scope
Only the four configs with committed dumps are recomputed here. The other seven
configurations remain aggregate-only in
[`results/loso_aggregate_11method.csv`](../../results/loso_aggregate_11method.csv),
as stated in the paper. At n = 5 folds the exact permutation floor is 0.0625, so
effect size, fold consistency, and bootstrap CIs are the primary evidence rather
than p < 0.05.
