#!/usr/bin/env python3
"""Control for the calibration claim: can one scalar pair do what KD does?

The paper's surviving claim is that KD improves calibration. The matching
control is the calibration analogue of a threshold sweep: fit a Platt scaling
(or a single bias) on the sessions that are already in the training split for a
given fold, apply it to the held-out session, and see how much of the
miscalibration that removes.

Reports, per fold: baseline reliability, baseline after a cross-fold-fitted bias
and after cross-fold Platt scaling, and each KD model's reliability. Then
applies the same recalibration to every model to test whether KD retains any
advantage once both are calibrated.

Usage:  python scripts/recalibration_control.py
"""
import glob, os
import numpy as np
from scipy import stats, optimize

P = os.path.join(os.path.dirname(__file__), "..", "results", "predictions_loso")
NAME = {"baseline_binary_v1": "Baseline", "kd_logit_unw": "LogitKD",
        "kd_logit_unw_immune": "LogitKD_TFD",
        "kd_hybrid_logit_cosine_immune": "Hybrid_TFD"}
ORDER = ["Baseline", "LogitKD", "LogitKD_TFD", "Hybrid_TFD"]
FIT_N = 200_000

L = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))


def reliability(p, y, nb=10):
    e = np.linspace(0, 1, nb + 1); a = 0.0
    for i in range(nb):
        m = (p > e[i]) & (p <= e[i + 1]) if i else (p >= e[i]) & (p <= e[i + 1])
        if m.sum():
            a += m.sum() * (p[m].mean() - y[m].mean()) ** 2
    return a / len(y)


def nll(v, z, y):
    q = np.clip(sg(v[0] * z + v[1]), 1e-9, 1 - 1e-9)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()


def main():
    D = {NAME[os.path.basename(f)[:-4]]: np.load(f) for f in glob.glob(os.path.join(P, "*.npz"))}
    if "Baseline" not in D:
        raise SystemExit(f"no baseline npz in {P}")
    S = sorted({k.rsplit("_", 1)[0] for k in D["Baseline"].files})
    g = lambda n, s: (D[n][s + "_probs"].astype(float), D[n][s + "_labels"].astype(int))
    rng = np.random.default_rng(0)

    def fit(n, held):
        z, y = [], []
        for s in S:
            if s == held:
                continue
            p2, y2 = g(n, s)
            k = rng.choice(len(p2), min(FIT_N, len(p2)), replace=False)
            z.append(L(p2[k])); y.append(y2[k])
        z, y = np.concatenate(z), np.concatenate(y)
        b = optimize.minimize(lambda v: nll([1.0, v[0]], z, y), [0.0], method="Nelder-Mead").x[0]
        ab = optimize.minimize(lambda v: nll(v, z, y), [1.0, 0.0], method="Nelder-Mead").x
        return b, ab

    print("Brier reliability, lower is better. Recalibration fitted on the other four")
    print("sessions, which are training data for the held-out fold.\n")
    print(f"  {'held-out fold':18s} {'baseline':>9} {'+bias':>8} {'+Platt':>8} "
          + "".join(f"{n:>12}" for n in ORDER[1:]))
    rows = []
    for s in S:
        p, y = g("Baseline", s); z = L(p)
        b, ab = fit("Baseline", s)
        r = [reliability(p, y), reliability(sg(z + b), y), reliability(sg(ab[0] * z + ab[1]), y)]
        r += [reliability(*g(n, s)) for n in ORDER[1:]]
        rows.append(r)
        print(f"  {s:18s} " + " ".join(f"{v:8.4f}" for v in r[:3])
              + "".join(f"{v:12.4f}" for v in r[3:]))
    A = np.array(rows)
    print(f"  {'MEAN':18s} " + " ".join(f"{v:8.4f}" for v in A.mean(0)[:3])
          + "".join(f"{v:12.4f}" for v in A.mean(0)[3:]))
    print()
    for k, n in enumerate(ORDER[1:], start=3):
        for j, lbl in ((1, "cross-fold bias"), (2, "cross-fold Platt")):
            d = A[:, k] - A[:, j]
            p = stats.ttest_rel(A[:, k], A[:, j])[1]
            who = "KD better" if d.mean() < 0 else "recalibration better"
            print(f"  {n:12s} vs baseline + {lbl:17s}: diff {d.mean():+.4f}  p={p:.4f}  -> {who}")

    print("\nAfter recalibrating every model the same way, does KD keep any advantage?")
    out = {}
    for n in ORDER:
        v = []
        for s in S:
            _, ab = fit(n, s)
            p, y = g(n, s)
            v.append(reliability(sg(ab[0] * L(p) + ab[1]), y))
        out[n] = np.array(v)
        print(f"  {n:12s} reliability {out[n].mean():.5f}")
    for n in ORDER[1:]:
        d = out[n] - out["Baseline"]
        print(f"  {n:12s} vs recalibrated baseline: diff {d.mean():+.5f}  "
              f"p={stats.ttest_rel(out[n], out['Baseline'])[1]:.4f}")


if __name__ == "__main__":
    main()
