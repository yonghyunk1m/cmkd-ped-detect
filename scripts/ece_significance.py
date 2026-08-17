#!/usr/bin/env python3
"""Per-fold ECE and Brier decomposition, with paired tests against the baseline.

The calibration result is the one quantity in the paper that a threshold change
cannot produce: macro accuracy, Ped accuracy and F1 all lie on a single
operating-point line (R^2 = 0.996 across the eleven configurations), so they
cannot separate the methods. ECE and the Brier reliability term can, because a
monotone reparameterization of the scores leaves both unchanged by construction.

verify_paper_numbers.py already computes ECE per fold but only prints the mean.
This prints the per-fold values and runs the paired test, which is what the
headline claim should rest on.

Needs the same prediction dumps: results/predictions_loso/<config>.npz

Usage:  python scripts/ece_significance.py
"""
import glob, os
import numpy as np
from scipy import stats

PRED = os.path.join(os.path.dirname(__file__), "..", "results", "predictions_loso")
NAME = {"baseline_binary_v1": "Baseline", "kd_logit_unw": "LogitKD",
        "kd_logit_unw_immune": "LogitKD_TFD",
        "kd_hybrid_logit_cosine_immune": "Hybrid_TFD"}


def sessions(d):
    return sorted({k.rsplit("_", 1)[0] for k in d.files})


def ece(p, y, nb=10):
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
        if m.sum():
            rel += m.sum() * (p[m].mean() - y[m].mean()) ** 2
            res += m.sum() * (y[m].mean() - yb) ** 2
    return rel / N, res / N


def per_fold(d, fn):
    return np.array([fn(d[f"{s}_probs"].astype(float), d[f"{s}_labels"].astype(int))
                     for s in sessions(d)])


def main():
    data = {}
    for path in sorted(glob.glob(os.path.join(PRED, "*.npz"))):
        cfg = os.path.basename(path)[:-4]
        data[NAME.get(cfg, cfg)] = np.load(path)
    if "Baseline" not in data:
        raise SystemExit(f"no baseline npz in {PRED}")

    folds = sessions(data["Baseline"])
    E = {n: per_fold(d, ece) for n, d in data.items()}
    R = {n: per_fold(d, lambda p, y: brier(p, y)[0]) for n, d in data.items()}
    S = {n: per_fold(d, lambda p, y: brier(p, y)[1]) for n, d in data.items()}

    print("Per-fold ECE (lower is better)")
    print("  " + "fold".ljust(18) + "".join(n.rjust(14) for n in E))
    for i, f in enumerate(folds):
        print("  " + f.ljust(18) + "".join(f"{E[n][i]:14.4f}" for n in E))
    print("  " + "mean".ljust(18) + "".join(f"{E[n].mean():14.4f}" for n in E))

    print("\nPaired t-test vs. Baseline (n=%d folds)" % len(folds))
    print(f"  {'config':14s} {'metric':22s} {'mean diff':>10} {'d(pop)':>8} {'p':>8}  folds improved")
    for n in E:
        if n == "Baseline":
            continue
        for label, M, lower_better in (("ECE", E, True),
                                       ("Brier reliability", R, True),
                                       ("Brier resolution", S, False)):
            diff = M[n] - M["Baseline"]
            p = stats.ttest_rel(M[n], M["Baseline"])[1]
            d = diff.mean() / diff.std(ddof=0) if diff.std(ddof=0) > 0 else float("nan")
            better = int((diff < 0).sum() if lower_better else (diff > 0).sum())
            print(f"  {n:14s} {label:22s} {diff.mean():+10.4f} {d:+8.2f} {p:8.4f}  {better}/{len(folds)}")

    print("\nResolution is the discrimination term. If it does not move, KD changed")
    print("nothing about the ranking, and the ECE gain cannot be a threshold effect.")


if __name__ == "__main__":
    main()
