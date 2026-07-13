"""Multi-metric paired t-tests for completed LOSO methods.

Per-fold values are extracted from training logs (test/* metrics).
Computes paired t-test + Cohen's d for macro, F1, PR-AUC.
"""
import numpy as np
from scipy import stats

# Per-fold test metrics (extracted from training logs).
# Order: [Session_5242023, Session_6012023, Session_6072023, Session_6212023, Session_6282023]
PER_FOLD = {
    "baseline_binary_v1": {
        # 524 from KD_EXPERIMENT_SUMMARY.md; 601 derived from aggregate (n=3 mean
        # 0.2605, n=5 mean 0.2864 with known 524/607/621/628); 607/621/628 from logs.
        "macro": [0.6995, 0.7366, 0.7565, 0.7259, 0.7266],
        "f1":    [0.2807, 0.1721, 0.3305, 0.3201, 0.3287],
    },
    "kd_hybrid_logit_cosine_immune": {  # = Hybrid + TFD-Label
        "macro": [0.7035, 0.7482, 0.7579, 0.7320, 0.7378],
        "f1":    [0.3128, 0.2336, 0.3502, 0.3382, 0.3537],
    },
}


def paired_test(a, b, label_a="A", label_b="B"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = a - b
    n = len(diffs)
    mean_d = diffs.mean()
    std_d = diffs.std(ddof=1)
    t = mean_d / (std_d / np.sqrt(n))
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    d = mean_d / std_d  # Cohen's d for paired samples
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {label_a} vs {label_b}: mean_diff={mean_d:+.4f}  "
          f"t={t:+.3f}  p={p:.4f} {sig}  d={d:+.3f}")
    return mean_d, t, p, d


def main():
    bl = PER_FOLD["baseline_binary_v1"]
    kd = PER_FOLD["kd_hybrid_logit_cosine_immune"]
    print("=" * 60)
    print("Paired t-tests: Hybrid+TFD-Label vs Baseline (n=5 folds)")
    print("=" * 60)

    for metric in ("macro", "f1"):
        print(f"\n[{metric.upper()}]")
        bl_v = np.array(bl[metric])
        kd_v = np.array(kd[metric])
        print(f"  Baseline:   mean={bl_v.mean():.4f}  std={bl_v.std(ddof=1):.4f}  per-fold={bl_v}")
        print(f"  Hybrid+TFD: mean={kd_v.mean():.4f}  std={kd_v.std(ddof=1):.4f}  per-fold={kd_v}")
        paired_test(kd_v, bl_v, "Hybrid+TFD", "Baseline")


if __name__ == "__main__":
    main()
