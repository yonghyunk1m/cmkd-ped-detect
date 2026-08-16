#!/usr/bin/env python3
"""Verify the paper's LOSO table from committed per-fold metrics.

Unlike scripts/verify_paper_numbers.py, this needs no prediction dumps: it reads
results/loso_per_fold_wandb.csv, the per-fold test metrics recorded by the
training runs themselves, and cross-checks them against the aggregate written
later by scripts/aggregate_loso.py (results/loso_aggregate_11method.csv).

The two files come from independent paths -- one logged during training, the
other recomputed afterwards by reloading each checkpoint -- so agreement is a
genuine provenance check rather than a restatement.

It also verifies that every fold used its own LOSO teacher, which is what rules
out the held-out session leaking into the teacher head.

Coverage is partial: some folds were run without logging and are absent. The
script reports coverage explicitly and only compares means where all 5 folds
are present.

Usage:  python scripts/verify_from_per_fold.py
Requires: numpy, pandas, scipy.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
SESS = ["Session_5242023", "Session_6012023", "Session_6072023",
        "Session_6212023", "Session_6282023"]


def main():
    pf = pd.read_csv(os.path.join(RES, "loso_per_fold_wandb.csv"))
    ag = pd.read_csv(os.path.join(RES, "loso_aggregate_11method.csv")).set_index("method")

    print("=" * 78)
    print("1. Per-fold coverage")
    print("=" * 78)
    cov = pf.groupby("method").size()
    for m in ag.index:
        print(f"  {m:32s} {cov.get(m, 0)}/5 folds logged")

    print()
    print("=" * 78)
    print("2. Teacher checkpoint per fold (leakage check)")
    print("=" * 78)
    t = pf.dropna(subset=["teacher_ckpt_dir"])
    t = t[t["teacher_ckpt_dir"].astype(str).str.len() > 0]
    bad = t[t["teacher_ckpt_dir"] != t["session"]]
    print(f"  runs with a recorded teacher path : {len(t)}")
    print(f"  runs whose teacher is NOT its own fold's LOSO teacher: {len(bad)}")
    if len(bad):
        print("  !! these folds shared a teacher trained on the held-out session:")
        for _, r in bad.iterrows():
            print(f"     {r['method']} / {r['session']} -> {r['teacher_ckpt_dir']}")
    else:
        print("  OK: every logged KD run used work_dir/teacher_video_only/<its own fold>/,")
        print("      i.e. a teacher head fit on the four training sessions only.")

    print()
    print("=" * 78)
    print("3. Per-fold mean vs. the committed aggregate (5-fold methods only)")
    print("=" * 78)
    full = [m for m in ag.index if cov.get(m, 0) == 5]
    print(f"{'method':32s} {'per-fold mean':>13} {'aggregate':>10} {'delta':>8}")
    for m in full:
        v = pf[pf.method == m]["macro_accuracy"].values
        a = float(ag.loc[m, "macro_mean"])
        print(f"{m:32s} {v.mean():13.4f} {a:10.4f} {v.mean()-a:+8.4f}")
    skipped = [m for m in ag.index if m not in full]
    if skipped:
        print(f"\n  not compared (partial coverage): {', '.join(skipped)}")

    print()
    print("=" * 78)
    print("4. Significance, where both baseline and method have all 5 folds")
    print("=" * 78)
    if "baseline_binary_v1" not in full:
        print("  baseline has only "
              f"{cov.get('baseline_binary_v1', 0)}/5 folds logged, so d and p cannot be")
        print("  recomputed here. Use scripts/verify_paper_numbers.py with the")
        print("  prediction dumps for the significance numbers in Tab. 1.")
        return
    bl = pf[pf.method == "baseline_binary_v1"].set_index("session").loc[SESS, "macro_accuracy"].values
    for m in full:
        if m == "baseline_binary_v1":
            continue
        v = pf[pf.method == m].set_index("session").loc[SESS, "macro_accuracy"].values
        d = (v - bl).mean() / (v - bl).std(ddof=0)
        p = stats.ttest_rel(v, bl)[1]
        print(f"  {m:32s} d={d:+.2f}  p={p:.3f}")


if __name__ == "__main__":
    main()
