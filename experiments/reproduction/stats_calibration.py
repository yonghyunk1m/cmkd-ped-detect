"""
Small-sample statistical strengthening of the paper's CALIBRATION claims,
computed from the committed per-fold calibration results
(loso_per_fold_calibration.csv, 4 configs x 5 LOSO folds, same sessions).

For each config vs. Baseline and each metric we report, on the 5 paired folds:
  - mean improvement (Baseline - config; positive = better for ECE/reliability),
  - EXACT paired sign-flip permutation p (all 2^5 = 32 flips; valid at n=5,
    no normality assumption), two-sided,
  - paired Cohen's d (population std, matching the paper),
  - folds improved (out of 5),
  - percentile bootstrap 95% CI of the mean improvement (coarse at n=5),
  - Holm-Bonferroni-adjusted permutation p across the primary family.

resolution (discrimination) is included as a control: it should NOT move.
This does not manufacture power; it replaces the t-test with an exact test and
adds a multiple-comparison correction, which is what a reviewer expects at n=5.
"""
import csv, collections, itertools
import numpy as np

rows = list(csv.DictReader(open('loso_per_fold_calibration.csv')))
sessions = sorted({r['session'] for r in rows})
def vec(method, key):
    d = {r['session']: float(r[key]) for r in rows if r['method'] == method}
    return np.array([d[s] for s in sessions])

configs = ['LogitKD', 'LogitKD_TFD', 'Hybrid_TFD']
metrics = [('ece', 'ECE'), ('brier_reliability', 'reliability'), ('brier_resolution', 'resolution(control)')]
base = {k: vec('Baseline', k) for k, _ in metrics}


def exact_perm_p(diffs):
    obs = diffs.mean(); n = len(diffs); cnt = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs((np.array(signs) * diffs).mean()) >= abs(obs) - 1e-12:
            cnt += 1
    return cnt / (2 ** n)


def boot_ci(diffs, B=20000, seed=0):
    r = np.random.default_rng(seed); n = len(diffs)
    means = np.array([diffs[r.integers(0, n, n)].mean() for _ in range(B)])
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


print(f"n_folds={len(sessions)} sessions={[s.split('_')[-1] for s in sessions]}\n")
print(f"{'config':>12} {'metric':>20} {'base':>7} {'cfg':>7} {'improv':>7} {'d':>6} {'perm_p':>7} {'impr/5':>6} {'95% CI':>18}")
primary = []  # (label, perm_p) for Holm over ECE + reliability
for c in configs:
    for key, name in metrics:
        b = base[key]; x = vec(c, key)
        impr = b - x                      # positive = config better (lower error)
        d = impr.mean() / (impr.std() + 1e-12)   # population std (paper convention)
        p = exact_perm_p(impr)
        lo, hi = boot_ci(impr)
        n_impr = int((impr > 0).sum())
        print(f"{c:>12} {name:>20} {b.mean():>7.3f} {x.mean():>7.3f} {impr.mean():>7.3f} "
              f"{d:>6.2f} {p:>7.3f} {n_impr:>4}/5 [{lo:>7.4f},{hi:>7.4f}]")
        if key in ('ece', 'brier_reliability'):
            primary.append((f"{c}:{name}", p))
    print()

# Holm-Bonferroni across the primary family (ECE + reliability, 3 configs = 6 tests)
print("Holm-Bonferroni across the primary family (ECE & reliability, 6 tests):")
order = sorted(primary, key=lambda t: t[1]); m = len(order); prev = 0.0
for i, (lab, p) in enumerate(order):
    adj = min(1.0, max(prev, (m - i) * p)); prev = adj
    print(f"  {lab:>26}  raw p={p:.3f}  Holm p={adj:.3f}  {'sig@.05' if adj < 0.05 else 'ns'}")

print("\nNotes:")
print(" - Exact permutation p replaces the normality-assuming t-test at n=5.")
print(" - resolution (discrimination) should stay ns -> KD adds no ranking info.")
print(" - Minimum possible two-sided exact p at n=5 is 2/32 = 0.0625, so with only")
print("   5 folds NO single test can reach p<.05 by permutation; we therefore lean on")
print("   effect size + 5/5-fold consistency, and report this honestly.")
