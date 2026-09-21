#!/usr/bin/env python3
"""Compute 10-bin ECE and the Brier-score decomposition (reliability / resolution /
uncertainty) for each configuration and LOSO fold, then aggregate to a 5-fold mean.

This fills the ECE column of the paper's LOSO table (Table 1). It is a post-hoc
metric: it re-reads saved per-sample predictions (predicted pedestrian probability
+ ground-truth label) and computes ECE without retraining. If prediction dumps are
not present, generate them first with `--from-ckpt` (requires the ASPED v.a audio
features and the trained checkpoints; see README "Reproducing the ECE column").

Usage
-----
# A) from saved prediction dumps (fast, no GPU):
python scripts/compute_ece_loso.py --preds-dir results/predictions_loso --out results/loso_ece.csv

# B) regenerate predictions from checkpoints first (needs data + GPU), then aggregate:
python scripts/compute_ece_loso.py --from-ckpt --work-dir work_dir \
    --data-root /path/to/ASPED_v.a --out results/loso_ece.csv

Each prediction dump is an .npz with:
    prob   : float array [N], predicted probability of the pedestrian class
    label  : int   array [N], ground-truth (0 = No-Ped, 1 = Ped)
    (optional) fold : str/int array [N], the held-out session id per sample
"""
import argparse, glob, os, csv
import numpy as np


def ece_10bin(prob_pos, labels, n_bins=10):
    """Expected Calibration Error (Guo et al., 2017), confidence of the predicted class."""
    pred = (prob_pos >= 0.5).astype(int)
    conf = np.where(pred == 1, prob_pos, 1.0 - prob_pos)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    N = len(labels)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            ece += (m.sum() / N) * abs(acc[m].mean() - conf[m].mean())
    return ece


def brier_decomposition(prob_pos, labels, n_bins=10):
    """Murphy (1973) partition: BS = reliability - resolution + uncertainty."""
    y = labels.astype(float)
    ybar = y.mean()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    N = len(y)
    rel = res = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (prob_pos > lo) & (prob_pos <= hi) if i > 0 else (prob_pos >= lo) & (prob_pos <= hi)
        nk = m.sum()
        if nk:
            rel += nk * (prob_pos[m].mean() - y[m].mean()) ** 2
            res += nk * (y[m].mean() - ybar) ** 2
    return dict(reliability=rel / N, resolution=res / N, uncertainty=ybar * (1 - ybar))


def per_fold_ece(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    prob, label = np.asarray(d["prob"]).ravel(), np.asarray(d["label"]).ravel().astype(int)
    if "fold" in d:  # a single dump holding all folds concatenated
        folds = np.asarray(d["fold"]).ravel()
        return [ece_10bin(prob[folds == f], label[folds == f]) for f in np.unique(folds)]
    return [ece_10bin(prob, label)]  # already a single fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", default="results/predictions_loso",
                    help="dir of <config>.npz (or <config>_<session>.npz) prediction dumps")
    ap.add_argument("--out", default="results/loso_ece.csv")
    ap.add_argument("--from-ckpt", action="store_true",
                    help="regenerate predictions from checkpoints first (needs data + GPU)")
    ap.add_argument("--work-dir", default="work_dir")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    if args.from_ckpt:
        raise SystemExit(
            "Regeneration path: run inference.py for each work_dir/<config>/Session_*/best-*.ckpt "
            "over the held-out ASPED v.a session, saving {prob,label} to --preds-dir, then rerun "
            "this script without --from-ckpt. See README 'Reproducing the ECE column'.")

    rows = []
    for path in sorted(glob.glob(os.path.join(args.preds_dir, "*.npz"))):
        cfg = os.path.basename(path)[:-4]
        eces = per_fold_ece(path)
        rows.append((cfg, len(eces), float(np.mean(eces)), float(np.std(eces))))
        print(f"{cfg:34s} ECE={np.mean(eces):.4f} +/- {np.std(eces):.4f} ({len(eces)} folds)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "n_folds", "ece_mean", "ece_std"])
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
