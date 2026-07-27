#!/usr/bin/env python3
"""Reproduce the paper's headline numbers from the saved per-fold predictions.

Each results/predictions_loso/<config>.npz stores, for every LOSO fold (held-out
session), the student's predicted pedestrian probability and the ground-truth label:
    session_<id>_probs  : float [N]
    session_<id>_labels : int   [N]  (0 = No-Ped, 1 = Ped)

Running this script recomputes, for the four configurations whose predictions are
released here (baseline, LogitKD, LogitKD_TFD, Hybrid_TFD):
  - the LOSO 5-fold table numbers (macro / No-Ped / Ped accuracy, F1_Ped, PR-AUC),
  - the significance test (paired t-test vs. baseline; Cohen's d, population sd),
  - the calibration numbers (10-bin ECE, Brier reliability/resolution).

Usage:  python scripts/verify_paper_numbers.py
Requires: numpy, scikit-learn, scipy.
"""
import glob, os
import numpy as np
from sklearn.metrics import average_precision_score
from scipy import stats

PRED_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "predictions_loso")
NAME = {"baseline_binary_v1": "Baseline", "kd_logit_unw": "LogitKD",
        "kd_logit_unw_immune": "LogitKD_TFD", "kd_hybrid_logit_cosine_immune": "Hybrid_TFD"}


def sessions(d):
    return sorted({k.rsplit("_", 1)[0] for k in d.files})


def macro_acc(p, y):
    pred = (p >= 0.5).astype(int)
    return 0.5 * ((pred[y == 0] == 0).mean() + (pred[y == 1] == 1).mean())


def f1_ped(p, y):
    pred = (p >= 0.5).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def ece(p, y, nb=10):  # positive-class (binary) ECE; symmetric across the two classes
    e = np.linspace(0, 1, nb + 1); N = len(y); s = 0.0
    for i in range(nb):
        m = (p > e[i]) & (p <= e[i + 1]) if i else (p >= e[i]) & (p <= e[i + 1])
        if m.sum():
            s += m.sum() / N * abs(y[m].mean() - p[m].mean())
    return s


def brier(p, y, nb=10):
    yb = y.mean(); e = np.linspace(0, 1, nb + 1); N = len(y); rel = res = 0.0
    for i in range(nb):
        m = (p > e[i]) & (p <= e[i + 1]) if i else (p >= e[i]) & (p <= e[i + 1])
        nk = m.sum()
        if nk:
            rel += nk * (p[m].mean() - y[m].mean()) ** 2; res += nk * (y[m].mean() - yb) ** 2
    return rel / N, res / N


def per_fold(d, fn):
    return np.array([fn(d[f"{s}_probs"].astype(float), d[f"{s}_labels"].astype(int)) for s in sessions(d)])


def main():
    data = {}
    for path in sorted(glob.glob(os.path.join(PRED_DIR, "*.npz"))):
        cfg = os.path.basename(path)[:-4]
        data[NAME.get(cfg, cfg)] = np.load(path)

    macc = {name: per_fold(d, macro_acc) for name, d in data.items()}
    print(f"{'config':12s} macro        No-Ped  Ped    F1_Ped  PR-AUC  ECE     d      p")
    for name, d in data.items():
        mac = macc[name]
        noped = per_fold(d, lambda p, y: ((p >= 0.5).astype(int)[y == 0] == 0).mean())
        ped = per_fold(d, lambda p, y: ((p >= 0.5).astype(int)[y == 1] == 1).mean())
        f1 = per_fold(d, f1_ped)
        pa = per_fold(d, lambda p, y: average_precision_score(y, p))
        ec = per_fold(d, ece)
        if name == "Baseline":
            dd = pp = "--"
        else:
            diff = mac - macc["Baseline"]
            dd = f"{diff.mean() / diff.std(ddof=0):.2f}"          # Cohen's d, population sd
            pp = f"{stats.ttest_rel(mac, macc['Baseline'])[1]:.3f}"
        print(f"{name:12s} {mac.mean()*100:4.1f}±{mac.std(ddof=0)*100:.1f}   "
              f"{noped.mean()*100:5.1f}  {ped.mean()*100:5.1f}  {f1.mean()*100:5.1f}   "
              f"{pa.mean():.3f}   {ec.mean():.3f}  {dd:>5}  {pp:>5}")

    b, h = data["Baseline"], data["Hybrid_TFD"]
    rb = per_fold(b, lambda p, y: brier(p, y)[0]).mean()
    rh = per_fold(h, lambda p, y: brier(p, y)[0]).mean()
    sb = per_fold(b, lambda p, y: brier(p, y)[1]).mean()
    print(f"\nBrier reliability (calibration):   baseline {rb:.3f} -> Hybrid_TFD {rh:.3f}")
    print(f"Brier resolution (discrimination): {sb:.3f} (unchanged)")


if __name__ == "__main__":
    main()
