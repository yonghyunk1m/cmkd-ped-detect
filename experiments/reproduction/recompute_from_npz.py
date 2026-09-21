"""
Recompute per-fold LOSO metrics for the 4 key configs directly from the
prediction dumps (session_<id>_probs / _labels), verify against the paper's
Table 1, and run small-sample paired tests (exact permutation + bootstrap +
Holm) for the headline comparisons vs Baseline.

This closes the reproducibility gap for the paper's central rows (Baseline,
LogitKD, LogitKD_TFD, Hybrid_TFD). The other 7 configs have no dumps and remain
aggregate-only.

Point NPZ_DIR at the folder holding the four .npz files.
"""
import os, glob, csv, itertools, collections
import numpy as np
from sklearn.metrics import average_precision_score

NPZ_DIR = os.environ.get("NPZ_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "predictions_loso"))
NAME = {"baseline_binary_v1": "Baseline", "kd_logit_unw": "LogitKD",
        "kd_logit_unw_tfd": "LogitKD_TFD", "kd_hybrid_logit_cosine_tfd": "Hybrid_TFD"}
ORDER = ["Baseline", "LogitKD", "LogitKD_TFD", "Hybrid_TFD"]


def ece10(p, y, nb=10):
    e = np.linspace(0, 1, nb + 1); conf = np.where(p >= .5, p, 1 - p); pred = (p >= .5).astype(int)
    acc = (pred == y).astype(float); s = 0.0
    for i in range(nb):
        m = (p > e[i]) & (p <= e[i+1]) if i else (p >= e[i]) & (p <= e[i+1])
        if m.sum(): s += m.mean() * abs(conf[m].mean() - acc[m].mean())
    return s
def reliab(p, y, nb=10):
    e = np.linspace(0, 1, nb + 1); n = len(y); s = 0.0
    for i in range(nb):
        m = (p > e[i]) & (p <= e[i+1]) if i else (p >= e[i]) & (p <= e[i+1])
        if m.sum(): s += m.sum() * (p[m].mean() - y[m].mean())**2
    return s / n
def fold_metrics(p, y):
    pred = (p >= .5).astype(int)
    ap = (pred[y == 1] == 1).mean(); an = (pred[y == 0] == 0).mean()
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = ap
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(macro=50*(ap+an), acc_ped=100*ap, acc_noped=100*an, f1=100*f1,
                prauc=average_precision_score(y, p), ece=ece10(p, y), rel=reliab(p, y),
                posratio=(pred == 1).mean())

# load
data = {}
for f in glob.glob(os.path.join(NPZ_DIR, "*.npz")):
    key = os.path.splitext(os.path.basename(f))[0]
    if key not in NAME: continue
    d = np.load(f, allow_pickle=True)
    sess = sorted({k.rsplit('_', 1)[0] for k in d.keys()})
    data[NAME[key]] = {s: (np.asarray(d[f"{s}_probs"]).astype(np.float64),
                           np.asarray(d[f"{s}_labels"]).astype(int)) for s in sess}

sessions = sorted(next(iter(data.values())).keys())
per = {c: {s: fold_metrics(*data[c][s]) for s in sessions} for c in data}

# ---- Table (5-fold means) vs paper ----
print("=== 5-fold means (recomputed from npz) ===")
print(f"{'config':>12} {'Macro':>6} {'No-Ped':>7} {'Ped':>6} {'F1':>6} {'PR-AUC':>7} {'ECE':>6} {'rel':>6} {'posr':>6}")
paper = {"Baseline": (72.9, 63.5, 82.3, 28.6, .328), "LogitKD": (73.6, 70.8, 76.4, 31.2, .332),
         "LogitKD_TFD": (73.7, 72.6, 74.7, 31.8, .330), "Hybrid_TFD": (73.6, 72.9, 74.3, 31.8, .331)}
for c in ORDER:
    M = lambda k: np.mean([per[c][s][k] for s in sessions])
    print(f"{c:>12} {M('macro'):>6.1f} {M('acc_noped'):>7.1f} {M('acc_ped'):>6.1f} {M('f1'):>6.1f} "
          f"{M('prauc'):>7.3f} {M('ece'):>6.3f} {M('rel'):>6.4f} {M('posratio'):>6.3f}   paper={paper[c]}")

# ---- paired tests vs Baseline ----
def exact_p(diffs):
    obs = diffs.mean(); n = len(diffs)
    return sum(abs((np.array(sg)*diffs).mean()) >= abs(obs)-1e-12 for sg in itertools.product([1,-1], repeat=n)) / 2**n
def bootci(diffs, B=20000, seed=0):
    r = np.random.default_rng(seed); n = len(diffs)
    m = [diffs[r.integers(0, n, n)].mean() for _ in range(B)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

base = {k: np.array([per["Baseline"][s][k] for s in sessions]) for k in ['macro','f1','prauc','ece','rel']}
print("\n=== paired vs Baseline (5 folds; macro/F1/PR-AUC higher better, ECE/rel improvement=base-cfg) ===")
print(f"{'config':>12} {'metric':>7} {'mean_diff':>9} {'d':>6} {'perm_p':>7} {'win/5':>6} {'95% CI':>20}")
macro_tests = []
for c in ["LogitKD", "LogitKD_TFD", "Hybrid_TFD"]:
    for k in ['macro', 'f1', 'prauc', 'ece', 'rel']:
        x = np.array([per[c][s][k] for s in sessions])
        diff = (base[k] - x) if k in ('ece', 'rel') else (x - base[k])   # positive = better
        d = diff.mean()/(diff.std()+1e-12); p = exact_p(diff); lo, hi = bootci(diff)
        win = int((diff > 0).sum())
        print(f"{c:>12} {k:>7} {diff.mean():>9.4f} {d:>6.2f} {p:>7.3f} {win:>4}/5 [{lo:>8.4f},{hi:>8.4f}]")
        if k == 'macro':
            macro_tests.append((f"{c}", p))
    print()

print("Holm-Bonferroni over the 3 macro comparisons vs Baseline:")
order = sorted(macro_tests, key=lambda t: t[1]); m = len(order); prev = 0
for i, (lab, p) in enumerate(order):
    adj = min(1, max(prev, (m-i)*p)); prev = adj
    print(f"  {lab:>12}  raw p={p:.3f}  Holm p={adj:.3f}")

# ---- save per-fold CSV ----
with open("loso_per_fold_from_npz.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method", "session", "macro", "acc_noped", "acc_ped", "f1_ped", "prauc", "ece", "brier_reliability", "pos_ratio"])
    for c in ORDER:
        for s in sessions:
            r = per[c][s]
            w.writerow([c, s, f"{r['macro']:.4f}", f"{r['acc_noped']:.4f}", f"{r['acc_ped']:.4f}",
                        f"{r['f1']:.4f}", f"{r['prauc']:.5f}", f"{r['ece']:.5f}", f"{r['rel']:.6f}", f"{r['posratio']:.5f}"])
print("\nwrote loso_per_fold_from_npz.csv")
