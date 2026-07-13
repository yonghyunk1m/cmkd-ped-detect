"""Post-hoc threshold ablation — defends TFD-Label against the "just threshold tuning" attack.

For each LOSO fold, sweeps decision threshold on baseline predictions to find the
threshold that maximizes macro accuracy. Then compares:
  (1) Baseline @ 0.5 (default)
  (2) Baseline @ best-per-fold threshold
  (3) Baseline @ best-LOSO-mean threshold (more conservative, no per-fold cherry-picking)
  (4) Hybrid+TFD-Label @ 0.5

If TFD-Label still beats baseline-with-tuned-threshold, we have evidence that the method
does more than re-locate the operating point.

Usage:
    python scripts/threshold_ablation.py
"""
import os
import sys
import glob
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.utils import get_train_test_splits
from data.dataset import ASPEDDataset
from models.seq2seq import ASPEDLightningModel
from models.kd_model import ASPEDKDModel

FEATURE_ROOT = "/media/ykim/Linux/ASPED_v1_npy"
WORK_DIR = "work_dir"
SESSIONS = [
    "Session_5242023", "Session_6012023", "Session_6072023",
    "Session_6212023", "Session_6282023",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLDS = np.arange(0.05, 0.96, 0.025)


def find_best_ckpt(exp_name, session):
    base = os.path.join(WORK_DIR, exp_name, session)
    if not os.path.isdir(base):
        return None
    ckpts = (glob.glob(os.path.join(base, "best-*.ckpt")) +
             glob.glob(os.path.join(base, "best-*", "*.ckpt")))
    return sorted(ckpts)[-1] if ckpts else None


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
def collect_probs(method, session, n_classes=2):
    cfg_path = f"configs/{method}.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    is_kd = (config.get("model", {}).get("kd") is not None and
             not method.startswith("baseline"))

    ckpt = find_best_ckpt(method, session)
    if ckpt is None:
        return None, None
    model = build_model(config, is_kd)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval().to(DEVICE)

    _, test_list = get_train_test_splits(
        feature_root_dir=FEATURE_ROOT,
        test_session=session,
        window_size_sec=10,
        test_stride_sec=10,
    )
    dataset = ASPEDDataset(test_list, task_type="classification", max_class=n_classes - 1)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    probs, labels = [], []
    for batch in loader:
        audio, y = batch[0], batch[-1]
        audio = audio.to(DEVICE)
        logits = model(audio).squeeze(-1)
        p = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        probs.append(p)
        labels.append(y.numpy().reshape(-1))
    del model
    torch.cuda.empty_cache()
    return np.concatenate(probs), np.concatenate(labels)


def macro_at_threshold(probs, labels, t):
    preds = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    noped = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ped = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 0.5 * (noped + ped), noped, ped


def f1_at_threshold(probs, labels, t):
    preds = (probs >= t).astype(int)
    return f1_score(labels, preds, zero_division=0)


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 70)
    print("Post-hoc threshold ablation: does Hybrid+TFD-Label beat tuned baseline?")
    print("=" * 70)

    # Collect baseline probs per fold
    baseline_probs = {}
    for s in SESSIONS:
        p, y = collect_probs("baseline_binary_v1", s)
        if p is None:
            print(f"  [skip] baseline missing {s}")
            continue
        baseline_probs[s] = (p, y)
        print(f"  [collected] baseline {s}: n={len(y)}")

    # Per-fold best threshold
    per_fold_best = {}
    for s, (p, y) in baseline_probs.items():
        macros = [(t, *macro_at_threshold(p, y, t)) for t in THRESHOLDS]
        t_star, m_star, _, _ = max(macros, key=lambda x: x[1])
        per_fold_best[s] = (t_star, m_star)
        print(f"  baseline {s}: t*={t_star:.3f}  macro*={m_star:.4f}")

    # Find LOSO-mean-optimal threshold (more honest — no per-fold cherry-picking)
    macros_per_t = []
    for t in THRESHOLDS:
        m_per_fold = [macro_at_threshold(p, y, t)[0] for p, y in baseline_probs.values()]
        macros_per_t.append((t, float(np.mean(m_per_fold))))
    t_loso_best, m_loso_best = max(macros_per_t, key=lambda x: x[1])
    print(f"\nBaseline LOSO-best single threshold: t={t_loso_best:.3f}  "
          f"5-fold mean macro={m_loso_best:.4f}")

    # Per-fold macro at LOSO-best threshold
    print(f"\nPer-fold macro at LOSO-best threshold {t_loso_best:.3f}:")
    bl_at_loso_best = []
    for s, (p, y) in baseline_probs.items():
        m, np_, p_ = macro_at_threshold(p, y, t_loso_best)
        bl_at_loso_best.append(m)
        print(f"  {s}: macro={m:.4f}  No_Ped={np_:.4f}  Ped={p_:.4f}")

    # Compare against Hybrid+TFD-Label (already evaluated, hardcoded for reference)
    hybrid_per_fold = {
        "Session_5242023": 0.7035,
        "Session_6012023": 0.7482,
        "Session_6072023": 0.7579,
        "Session_6212023": 0.7320,
        "Session_6282023": 0.7378,
    }
    hybrid_mean = np.mean(list(hybrid_per_fold.values()))

    print(f"\n{'=' * 70}")
    print("VERDICT")
    print("=" * 70)
    print(f"Baseline @ 0.5            : 5-fold mean macro = 0.7290 (paper)")
    print(f"Baseline @ best LOSO t*={t_loso_best:.3f}: 5-fold mean macro = "
          f"{m_loso_best:.4f}")
    print(f"Hybrid+TFD-Label @ 0.5    : 5-fold mean macro = {hybrid_mean:.4f}")
    print()
    gap_to_tuned = hybrid_mean - m_loso_best
    print(f"Hybrid+TFD-Label vs threshold-tuned baseline: {gap_to_tuned:+.4f}")
    if gap_to_tuned > 0:
        print(">>> Hybrid+TFD-Label STILL BEATS post-hoc tuned baseline.")
        print(">>> Method does more than threshold tuning.")
    else:
        print(">>> Threshold-tuned baseline matches/beats Hybrid+TFD-Label.")
        print(">>> Method's gain is largely operating-point shift (caveat: paper claims this).")


if __name__ == "__main__":
    main()
