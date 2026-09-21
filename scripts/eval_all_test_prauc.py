"""eval_all_test_prauc.py — Test evaluation with PR-AUC for all experiments.

Scans test data once (stride=10 only, no preload), then evaluates each model.
Reports: Macro Acc, Per-class Acc, F1, Precision, Recall, PR-AUC.

I/O safe: uses test_stride=10 (not 1), num_workers=0, no preloading.

Usage:
    ionice -c3 nice -n 19 python scripts/eval_all_test_prauc.py 2>&1 | tee /tmp/eval_prauc.txt
"""
import os
import sys
import glob
import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.utils import get_train_test_splits
from data.dataset import ASPEDDataset, ASPEDDatasetKD
from models.seq2seq import ASPEDLightningModel
from models.kd_model import ASPEDKDModel

WORK_DIR = "work_dir"
TEST_SESSION = "Session_5242023"
FEATURE_ROOT = "/path/to/ASPED_v.a"
EMBEDDING_DIR = "/path/to/teacher_embeddings"

EXPERIMENTS = {
    # name: (config_path, is_kd) -- the 11 configurations compared in the paper
    "baseline_binary_v1":            ("configs/baseline_binary_v1.yaml", False),
    "kd_logit_unw":                  ("configs/kd_logit_unw.yaml", True),
    "kd_logit_alpha03":              ("configs/kd_logit_alpha03.yaml", True),
    "kd_logit_lossnorm":             ("configs/kd_logit_lossnorm.yaml", True),
    "kd_logit_adjustment":           ("configs/kd_logit_adjustment.yaml", True),
    "kd_logit_focal_kd":             ("configs/kd_logit_focal_kd.yaml", True),
    "kd_cosine_unw":                 ("configs/kd_cosine_unw.yaml", True),
    "kd_crd_unw":                    ("configs/kd_crd_unw.yaml", True),
    "kd_rkd_unw":                    ("configs/kd_rkd_unw.yaml", True),
    "kd_logit_unw_tfd":           ("configs/kd_logit_unw_tfd.yaml", True),
    "kd_hybrid_logit_cosine_tfd": ("configs/kd_hybrid_logit_cosine_tfd.yaml", True),
    # exploratory variants live in configs/exploratory/ (not reported)
}


def find_best_ckpt(exp_name):
    exp_dir = os.path.join(WORK_DIR, exp_name)
    for base in [os.path.join(exp_dir, TEST_SESSION), exp_dir]:
        if not os.path.isdir(base):
            continue
        ckpts = glob.glob(os.path.join(base, "best-*.ckpt")) + \
                glob.glob(os.path.join(base, "best-*", "*.ckpt"))
        if ckpts:
            return sorted(ckpts)[-1]
    return None


def build_model(config, is_kd):
    mc = config["model"]
    # backbone config can be mc["backbone_name"] or mc["backbone"]["name"]
    backbone_cfg = mc.get("backbone", {})
    if isinstance(backbone_cfg, dict):
        bb_name = backbone_cfg.get("name", mc.get("backbone_name", "vggish"))
        bb_ft = backbone_cfg.get("finetune", mc.get("backbone_finetune", False))
    else:
        bb_name = mc.get("backbone_name", "vggish")
        bb_ft = mc.get("backbone_finetune", False)
    kwargs = dict(
        token_dim=mc.get("token_dim", 128),
        nhead=mc.get("nhead", 4),
        dropout=mc.get("dropout", 0.2),
        num_classes=mc["n_classes"],
        num_layers=mc.get("num_layers", 1),
        task_type=mc.get("task", "classification"),
        backbone_name=bb_name,
        backbone_finetune=bb_ft,
    )
    if is_kd:
        return ASPEDKDModel(kd_cfg=mc["kd"], **kwargs)
    else:
        return ASPEDLightningModel(**kwargs)


@torch.no_grad()
def evaluate_model(model, loader, device="cuda"):
    """Run inference and compute all metrics including PR-AUC."""
    model.eval().to(device)
    all_probs = []   # P(Ped_Present) per second
    all_labels = []  # GT per second (0 or 1)

    for batch in tqdm(loader, desc="  Eval", leave=False):
        if len(batch) == 2:
            audio, labels = batch
        else:
            audio, _emb, labels = batch  # discard teacher embedding at inference

        audio = audio.to(device)
        labels = labels.numpy()  # [B, 10]

        logits = model(audio)  # forward only needs audio

        # Binary: logits [B, 10, 1] → sigmoid → P(Ped)
        logits = logits.squeeze(-1)  # [B, 10]
        probs = torch.sigmoid(logits).cpu().numpy()  # [B, 10]

        all_probs.append(probs.reshape(-1))
        all_labels.append(labels.reshape(-1))

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)

    # Basic metrics
    n = len(all_labels)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    noped_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    ped_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    macro_acc = (noped_acc + ped_acc) / 2

    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # PR-AUC (Average Precision) for Ped_Present class
    pr_auc = average_precision_score(all_labels, all_probs)

    return {
        "macro_acc": macro_acc,
        "noped_acc": noped_acc,
        "ped_acc": ped_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_seconds": n,
    }


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    torch.set_float32_matmul_precision("high")

    # ---- Step 1: Scan data ONCE (stride=10 for test, I/O safe) ----
    print("Scanning test data...")
    _, test_list = get_train_test_splits(
        feature_root_dir=FEATURE_ROOT,
        test_session=TEST_SESSION,
        window_size_sec=10,
        test_stride_sec=10,
        embedding_dir=EMBEDDING_DIR,
    )
    print(f"Test: {len(test_list):,} windows, {len(test_list)*10:,} seconds")

    # Build dataset types
    base_ds = ASPEDDataset(test_list, task_type="classification", max_class=1)
    kd_ds = ASPEDDatasetKD(test_list, task_type="classification", max_class=1, strip_metadata=False)
    kd_ds_strip = ASPEDDatasetKD(test_list, task_type="classification", max_class=1, strip_metadata=True)

    base_loader = DataLoader(base_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    kd_loader = DataLoader(kd_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    kd_strip_loader = DataLoader(kd_ds_strip, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # ---- Step 2: Evaluate each experiment ----
    results = []

    # Print header
    hdr = f"{'Method':<35} {'Macro':>5} {'NoPed':>5} {'Ped':>5} | {'Prec':>5} {'Rec':>5} {'F1':>5} {'PR-AUC':>6} | {'TP':>8} {'FP':>8} {'TN':>8} {'FN':>8}"
    sep = "-" * len(hdr)
    print(f"\n{hdr}")
    print(sep)

    for exp_name, (config_path, is_kd) in sorted(EXPERIMENTS.items()):
        if not os.path.exists(config_path):
            continue

        ckpt_path = find_best_ckpt(exp_name)
        if ckpt_path is None:
            continue

        with open(config_path) as f:
            config = yaml.safe_load(f)

        try:
            model = build_model(config, is_kd)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model_state = model.state_dict()
            loaded = {k: v for k, v in ckpt["state_dict"].items()
                      if k in model_state and v.shape == model_state[k].shape}
            model.load_state_dict(loaded, strict=False)

            # Select loader
            if not is_kd:
                loader = base_loader
            else:
                strip = config["model"].get("kd", {}).get("strip_metadata", False)
                loader = kd_strip_loader if strip else kd_loader

            r = evaluate_model(model, loader)
            r["experiment"] = exp_name

            # Print result row
            print(f"{exp_name:<35} {r['macro_acc']*100:5.1f} {r['noped_acc']*100:5.1f} {r['ped_acc']*100:5.1f} | "
                  f"{r['precision']:5.3f} {r['recall']:5.3f} {r['f1']:5.3f} {r['pr_auc']:6.4f} | "
                  f"{r['tp']:>8} {r['fp']:>8} {r['tn']:>8} {r['fn']:>8}")

            results.append(r)
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"{exp_name:<35} [ERROR] {e}")
            import traceback
            traceback.print_exc()

    # ---- Step 3: Save ----
    if results:
        df = pd.DataFrame(results)
        out_path = os.path.join(WORK_DIR, "test_evaluation_prauc.csv")
        df.to_csv(out_path, index=False)
        print(f"\n{'='*60}")
        print(f"Saved to {out_path}")

        # Print sorted summary
        print(f"\n{'='*60}")
        print("SORTED BY MACRO ACCURACY:")
        print(f"{'='*60}")
        df_sorted = df.sort_values("macro_acc", ascending=False)
        for _, row in df_sorted.iterrows():
            print(f"  {row['experiment']:<35} Macro={row['macro_acc']*100:5.1f}  F1={row['f1']:.3f}  PR-AUC={row['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
