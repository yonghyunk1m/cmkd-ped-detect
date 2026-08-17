#!/usr/bin/env python3
"""Threshold ablation and the calibration defense, computed from prediction dumps.

The reviewer's first question is: PR-AUC is unchanged, so isn't every accuracy
gain reachable by moving the baseline's decision threshold? This answers it from
the saved probabilities alone -- no GPU, no features, no checkpoints.

It reports, per fold and in aggregate:
  1. the baseline at its best threshold (per-fold oracle, and one shared LOSO
     threshold, which is the only variant that could be used in deployment),
  2. how much of each KD method's macro gap that recovers,
  3. the best F1 the baseline reaches at any threshold,
  4. ECE and Brier reliability at the baseline's tuned threshold -- both are
     invariant under a monotone reparameterization, which is the point.

Usage:  python scripts/threshold_from_npz.py [--pred_dir results/predictions_loso]
"""
import argparse, glob, os
import numpy as np
from scipy import stats

NAME = {"baseline_binary_v1": "Baseline", "kd_logit_unw": "LogitKD",
        "kd_logit_unw_immune": "LogitKD_TFD",
        "kd_hybrid_logit_cosine_immune": "Hybrid_TFD"}
GRID = np.arange(0.05, 0.955, 0.005)


def sessions(d):
    return sorted({k.rsplit("_", 1)[0] for k in d.files})


def macro_at(p, y, t):
    q = p >= t
    return 0.5 * ((~q[y == 0]).mean() + q[y == 1].mean())


def f1_at(p, y, t):
    q = p >= t
    tp = (q & (y == 1)).sum(); fp = (q & (y == 0)).sum(); fn = ((~q) & (y == 1)).sum()
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def ece(p, y, nb=10):
    e = np.linspace(0, 1, nb + 1); s = 0.0
    for i in range(nb):
        m = (p > e[i]) & (p <= e[i + 1]) if i else (p >= e[i]) & (p <= e[i + 1])
        if m.sum():
            s += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default=os.path.join(
        os.path.dirname(__file__), "..", "results", "predictions_loso"))
    a = ap.parse_args()

    data = {}
    for path in sorted(glob.glob(os.path.join(a.pred_dir, "*.npz"))):
        cfg = os.path.basename(path)[:-4]
        data[NAME.get(cfg, cfg)] = np.load(path)
    if "Baseline" not in data:
        raise SystemExit(f"no baseline npz in {a.pred_dir}")
    folds = sessions(data["Baseline"])
    get = lambda n, s: (data[n][f"{s}_probs"].astype(float),
                        data[n][f"{s}_labels"].astype(int))

    # one shared threshold, chosen to maximise the 5-fold mean macro accuracy
    curve = [np.mean([macro_at(*get("Baseline", s), t) for s in folds]) for t in GRID]
    t_shared = float(GRID[int(np.argmax(curve))])

    print("=" * 74)
    print("1. Baseline macro accuracy at 0.5 vs. tuned thresholds")
    print("=" * 74)
    print(f"  {'fold':18s} {'@0.5':>8} {'@t* (oracle)':>14} {'t*':>6} {'@t_shared':>10}")
    b05, bor, bsh = [], [], []
    for s in folds:
        p, y = get("Baseline", s)
        m05 = macro_at(p, y, 0.5)
        ms = [macro_at(p, y, t) for t in GRID]
        j = int(np.argmax(ms))
        b05.append(m05); bor.append(ms[j]); bsh.append(macro_at(p, y, t_shared))
        print(f"  {s:18s} {m05:8.4f} {ms[j]:14.4f} {GRID[j]:6.3f} {bsh[-1]:10.4f}")
    b05, bor, bsh = map(np.array, (b05, bor, bsh))
    print(f"  {'mean':18s} {b05.mean():8.4f} {bor.mean():14.4f} {t_shared:6.3f} {bsh.mean():10.4f}")

    print()
    print("=" * 74)
    print("2. How much of each KD gap does thresholding the baseline recover?")
    print("=" * 74)
    for n in data:
        if n == "Baseline":
            continue
        k = np.array([macro_at(*get(n, s), 0.5) for s in folds])
        gap = k.mean() - b05.mean()
        rec_sh = (bsh.mean() - b05.mean()) / gap * 100 if gap else float("nan")
        rec_or = (bor.mean() - b05.mean()) / gap * 100 if gap else float("nan")
        print(f"  {n:14s} macro {k.mean():.4f}  gap {gap:+.4f}  "
              f"recovered: shared-t {rec_sh:5.1f}%   per-fold oracle {rec_or:5.1f}%")

    print()
    print("=" * 74)
    print("3. Best F1_Ped reachable by thresholding the baseline")
    print("=" * 74)
    bf = np.array([max(f1_at(*get("Baseline", s), t) for t in GRID) for s in folds])
    print(f"  Baseline @ best-F1 threshold (per-fold oracle): {bf.mean()*100:.1f}")
    for n in data:
        k = np.array([f1_at(*get(n, s), 0.5) for s in folds])
        print(f"  {n:14s} @0.5: {k.mean()*100:.1f}")
    print("  -> if the tuned baseline wins here, say so; the accuracy axis is not the claim.")

    print()
    print("=" * 74)
    print("4. The part thresholding cannot touch")
    print("=" * 74)
    e05 = np.array([ece(*get("Baseline", s)) for s in folds])
    print(f"  Baseline ECE @0.5            : {e05.mean():.4f}")
    print(f"  Baseline ECE @t_shared={t_shared:.3f}  : {e05.mean():.4f}  (identical by construction:")
    print("                                  a threshold does not change the probabilities)")
    for n in data:
        if n == "Baseline":
            continue
        e = np.array([ece(*get(n, s)) for s in folds])
        d = e - e05
        p = stats.ttest_rel(e, e05)[1]
        print(f"  {n:14s} ECE {e.mean():.4f}  diff {d.mean():+.4f}  "
              f"p={p:.4f}  improved {int((d<0).sum())}/{len(folds)} folds")


if __name__ == "__main__":
    main()
