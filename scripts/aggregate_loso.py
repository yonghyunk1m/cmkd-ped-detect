"""aggregate_loso.py — Aggregate LOSO results across 5 folds with statistical tests.

Scans work_dir for per-session checkpoints, evaluates each on its held-out
test set, and reports mean +/- std across folds with paired t-tests.

Usage:
    python scripts/aggregate_loso.py --methods baseline_binary_v1 kd_logit_unw kd_logit_unw_selective kd_logit_alpha03
    python scripts/aggregate_loso.py --all
"""
import os
import sys
import glob
import argparse
import yaml
import numpy as np
import pandas as pd
from scipy import stats

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.utils import get_train_test_splits
from data.dataset import ASPEDDataset, ASPEDDatasetKD
from models.seq2seq import ASPEDLightningModel
from models.kd_model import ASPEDKDModel

FEATURE_ROOT = "/media/ykim/Linux/ASPED_v1_npy"
EMBEDDING_DIR = "/media/ykim/Linux/ASPEDv1_VideoEmbeddings"
WORK_DIR = "work_dir"
SESSIONS = [
    "Session_5242023", "Session_6012023", "Session_6072023",
    "Session_6212023", "Session_6282023",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_best_ckpt(exp_name, session):
    base = os.path.join(WORK_DIR, exp_name, session)
    if not os.path.isdir(base):
        return None
    ckpts = glob.glob(os.path.join(base, "best-*.ckpt")) + \
            glob.glob(os.path.join(base, "best-*", "*.ckpt"))
    if not ckpts:
        return None
    # Pick the most recently modified ckpt. Previous alphabetical-sort logic
    # was order-dependent and could silently pick an older bug-era checkpoint
    # over a newer post-fix retrain (e.g. "best-epoch=00.ckpt" sorts after
    # "best-epoch=00-v1.ckpt" because '.' > '-' in ASCII, so the older file
    # would be selected). Sorting by mtime makes selection unambiguous and
    # always favours the most recent training run.
    return max(ckpts, key=lambda p: os.path.getmtime(p))


def build_model(config, is_kd):
    mc = config["model"]
    kwargs = dict(
        token_dim=mc.get("token_dim", 128),
        nhead=mc.get("nhead", 4),
        dropout=mc.get("dropout", 0.2),
        num_classes=mc["n_classes"],
        num_layers=mc.get("num_layers", 1),
        task_type=mc.get("task", "classification"),
        backbone_name=mc.get("backbone_name", "vggish"),
        backbone_finetune=mc.get("backbone_finetune", False),
    )
    if is_kd:
        return ASPEDKDModel(kd_cfg=mc["kd"], **kwargs)
    return ASPEDLightningModel(**kwargs)


@torch.no_grad()
def evaluate_fold(model, session, n_classes=2):
    model.eval().to(DEVICE)
    _, test_list = get_train_test_splits(
        feature_root_dir=FEATURE_ROOT,
        test_session=session,
        window_size_sec=10,
        test_stride_sec=10,
    )
    dataset = ASPEDDataset(test_list, task_type="classification", max_class=n_classes - 1)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    all_probs, all_labels = [], []
    for batch in loader:
        audio, labels = batch[0], batch[-1]
        audio = audio.to(DEVICE)
        logits = model(audio).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs.reshape(-1))
        all_labels.append(labels.numpy().reshape(-1))

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    noped_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    ped_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    macro = (noped_acc + ped_acc) / 2

    # NaN-safe PR-AUC: a model that diverged during training can produce NaN
    # sigmoid outputs on some test windows (numerical instability surviving
    # mid-training). Skip those samples for PR-AUC only; keep the threshold-
    # based macro/F1/precision/recall, since (probs >= 0.5) silently maps
    # NaN to False and the per-class accuracies remain meaningful.
    finite_mask = np.isfinite(all_probs)
    if finite_mask.all():
        prauc = average_precision_score(all_labels, all_probs)
        prauc_note = None
    elif finite_mask.sum() > 0:
        prauc = average_precision_score(all_labels[finite_mask],
                                        all_probs[finite_mask])
        n_nan = int((~finite_mask).sum())
        prauc_note = f"PR-AUC computed on {finite_mask.sum()}/{len(all_probs)} samples ({n_nan} NaN dropped)"
        print(f"    [warn] {prauc_note}")
    else:
        prauc = float("nan")
        prauc_note = "all probabilities NaN — PR-AUC unavailable"
        print(f"    [warn] {prauc_note}")

    return {
        "macro": macro,
        "noped": noped_acc,
        "ped": ped_acc,
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "prec": precision_score(all_labels, all_preds, zero_division=0),
        "rec": recall_score(all_labels, all_preds, zero_division=0),
        "prauc": prauc,
        "n": len(all_labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", default="work_dir/loso_aggregate.csv")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.all:
        methods = sorted([
            d for d in os.listdir(WORK_DIR)
            if os.path.isdir(os.path.join(WORK_DIR, d)) and d != "wandb"
        ])
    elif args.methods:
        methods = args.methods
    else:
        methods = ["baseline_binary_v1", "kd_logit_unw",
                    "kd_logit_unw_selective", "kd_logit_alpha03"]

    rows = []
    method_fold_macros = {}

    for method in methods:
        cfg_path = f"configs/{method}.yaml"
        if not os.path.exists(cfg_path):
            print(f"  [skip] {method}: no config")
            continue

        with open(cfg_path) as f:
            config = yaml.safe_load(f)
        is_kd = config.get("model", {}).get("kd") is not None and \
                not method.startswith("baseline")

        fold_results = []
        missing = []

        for session in SESSIONS:
            ckpt = find_best_ckpt(method, session)
            if ckpt is None:
                missing.append(session)
                continue
            model = build_model(config, is_kd)
            ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt_data["state_dict"], strict=False)
            result = evaluate_fold(model, session, config["model"]["n_classes"])
            result["session"] = session
            fold_results.append(result)
            del model
            torch.cuda.empty_cache()

        if not fold_results:
            print(f"  [skip] {method}: no checkpoints")
            continue

        if missing:
            print(f"  [warn] {method}: missing {len(missing)} folds: {missing}")

        df = pd.DataFrame(fold_results)
        macros = df["macro"].values
        method_fold_macros[method] = macros

        row = {"method": method, "n_folds": len(fold_results)}
        for col in ["macro", "noped", "ped", "f1", "prauc"]:
            vals = df[col].values
            row[f"{col}_mean"] = vals.mean()
            row[f"{col}_std"] = vals.std()
        rows.append(row)

        print(f"  {method}: macro={row['macro_mean']:.4f}+/-{row['macro_std']:.4f} "
              f"F1={row['f1_mean']:.4f}+/-{row['f1_std']:.4f} "
              f"PR-AUC={row['prauc_mean']:.4f}+/-{row['prauc_std']:.4f} "
              f"({len(fold_results)} folds)")

    result_df = pd.DataFrame(rows)

    # Paired t-tests vs baseline
    baseline_key = "baseline_binary_v1"
    if baseline_key in method_fold_macros and len(method_fold_macros[baseline_key]) >= 3:
        bl = method_fold_macros[baseline_key]
        print(f"\n{'='*60}")
        print(f"Paired t-tests vs {baseline_key} (n={len(bl)} folds)")
        print(f"{'='*60}")
        for method, macros in method_fold_macros.items():
            if method == baseline_key:
                continue
            n = min(len(bl), len(macros))
            if n < 3:
                continue
            t_stat, p_val = stats.ttest_rel(macros[:n], bl[:n])
            d = (macros[:n] - bl[:n]).mean() / (macros[:n] - bl[:n]).std() if (macros[:n] - bl[:n]).std() > 0 else 0
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {method:<40} t={t_stat:+.3f}  p={p_val:.4f} {sig}  d={d:.3f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
