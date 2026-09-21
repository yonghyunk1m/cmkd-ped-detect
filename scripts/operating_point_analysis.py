#!/usr/bin/env python
"""operating_point_analysis.py — threshold-free / matched-FPR comparison for the
four LOSO configurations with committed per-fold predictions.

For each configuration we load the per-session (per-fold) {prob, label} dumps in
results/predictions_loso/<config>.npz and, PER FOLD, compute:
  - ROC-AUC and PR-AUC (average precision),
  - pedestrian recall (TPR) at fixed false-positive rates via linear
    interpolation on that fold's ROC curve,
  - the operating point of the fixed 0.5 threshold (No-Ped acc, Ped recall, FPR).
Fold values are then averaged over the five folds (never pooled across folds,
so scores from different models are not mixed). We also print the per-fold
paired recall@FPR difference against the baseline, since a positive mean is not
the same as an improvement in every fold.

This reproduces the matched-FPR / ROC-AUC numbers reported in Section 5.1 and
verifies that the 0.5-threshold per-fold means match Table 1.

Usage:  python scripts/operating_point_analysis.py
Requires: numpy, scikit-learn.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

PRED_DIR = "results/predictions_loso"
CONFIGS = [
    ("Baseline",    "baseline_binary_v1"),
    ("LogitKD",     "kd_logit_unw"),
    ("LogitKD_TFD", "kd_logit_unw_tfd"),
    ("Hybrid_TFD",  "kd_hybrid_logit_cosine_tfd"),
]
FPRS = [0.05, 0.10, 0.20, 0.30, 0.365]


def load_folds(stem):
    d = np.load(f"{PRED_DIR}/{stem}.npz")
    sessions = sorted({k[:-6] for k in d.files if k.endswith("_probs")})
    return [(s,
             np.asarray(d[f"{s}_probs"]).astype(float),
             np.asarray(d[f"{s}_labels"]).astype(int)) for s in sessions]


def recall_at_fpr(y, p, target_fpr):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(target_fpr, fpr, tpr))


def main():
    data = {name: load_folds(stem) for name, stem in CONFIGS}

    print("== 0.5-threshold operating point (per fold, then averaged) ==")
    print(f"{'Config':<12}{'Macro':>7}{'NoPed':>7}{'PedRec':>7}{'FPR':>7}")
    for name, _ in CONFIGS:
        an = ap = fp = []
        an = [((p >= .5)[y == 0] == 0).mean() for _, p, y in data[name]]
        ap = [((p >= .5)[y == 1] == 1).mean() for _, p, y in data[name]]
        fp = [1 - a for a in an]
        print(f"{name:<12}{50*(np.mean(an)+np.mean(ap)):>7.1f}"
              f"{100*np.mean(an):>7.1f}{100*np.mean(ap):>7.1f}{np.mean(fp):>7.3f}")

    print("\n== threshold-free ranking + recall at matched FPR (5-fold means) ==")
    hdr = "  ".join(f"@{t:.3f}" for t in FPRS)
    print(f"{'Config':<12}{'ROC-AUC':>8}{'PR-AUC':>8}   {hdr}")
    for name, _ in CONFIGS:
        F = data[name]
        roc = np.mean([roc_auc_score(y, p) for _, p, y in F])
        pr = np.mean([average_precision_score(y, p) for _, p, y in F])
        rr = [np.mean([recall_at_fpr(y, p, t) for _, p, y in F]) for t in FPRS]
        print(f"{name:<12}{roc:>8.3f}{pr:>8.3f}   " + "  ".join(f"{r:>6.3f}" for r in rr))

    print("\n== per-fold paired recall@FPR delta vs baseline (%p) ==")
    bmap = {s: (p, y) for s, p, y in data["Baseline"]}
    for name, _ in CONFIGS[1:]:
        for t in [0.10, 0.20, 0.30]:
            diffs = [recall_at_fpr(y, p, t) - recall_at_fpr(bmap[s][1], bmap[s][0], t)
                     for s, p, y in data[name]]
            signs = "".join("+" if x > 0 else "-" for x in diffs)
            print(f"{name:<12} FPR={t:.2f}  mean {100*np.mean(diffs):+5.1f}  "
                  f"folds[{signs}] {[round(100*x, 1) for x in diffs]}")


if __name__ == "__main__":
    main()
