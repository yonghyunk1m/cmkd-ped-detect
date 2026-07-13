"""
eval_teacher.py — Standalone evaluation of the Mask2Former teacher MLP.

Loads pre-computed 258-dim video embeddings + GT labels and measures
binary pedestrian detection performance per session (LOSO style).

Usage:
    python eval_teacher.py
    python eval_teacher.py --teacher_ckpt <path>  --embed_dir <path>  --label_root <path>
"""

import os
import re
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Teacher MLP  (must match the checkpoint layout exactly)
# compress_mlp.0 : Linear(258→128)
# compress_mlp.1 : BatchNorm1d(128)
# classifier_exist : Linear(128→2)
# ─────────────────────────────────────────────────────────────────────────────

class TeacherMLP(nn.Module):
    def __init__(self, input_dim: int = 258):
        super().__init__()
        self.compress_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.classifier_exist = nn.Linear(128, 2)

    def forward(self, x):          # x: [N, input_dim]
        return self.classifier_exist(self.compress_mlp(x))  # [N, 2]


def load_teacher(ckpt_path: str, input_dim: int = 258, device: str = 'cpu') -> TeacherMLP:
    model = TeacherMLP(input_dim=input_dim).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        sd   = ckpt.get('state_dict', ckpt)
        model.load_state_dict(sd, strict=True)
        print(f"[Teacher] Loaded weights from {ckpt_path}")
    else:
        print("[Teacher] ⚠️  No checkpoint — using random weights (sanity check only).")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def iter_clips(embed_dir: str, label_root: str):
    """Yield (session, location, clip_id, recorder_idx, emb_tensor, binary_labels)."""
    pattern = re.compile(
        r'^(Session_\d+)_(\w+)_(\d+_6m_ellipse)_(\d+)_embed_258\.pt$'
    )
    for fname in sorted(os.listdir(embed_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        session, location, clip_stem, rec_idx = m.groups()
        rec_idx = int(rec_idx)

        # Label CSV:  label_root/{session}/{location}/Labels/{clip_id}.csv
        # clip_stem = "0001_6m_ellipse" → csv name = "0001.csv"
        clip_id  = clip_stem.split('_')[0]   # e.g. "0001"
        csv_path = os.path.join(label_root, session, location, 'Labels', f'{clip_id}.csv')
        if not os.path.exists(csv_path):
            continue

        col = f'recorder{rec_idx}_6m'
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if col not in df.columns:
            continue

        # Binary labels: 1 if count > 0
        raw_labels = df[col].values.astype(int)
        binary_labels = (raw_labels > 0).astype(int)

        # Load embeddings
        pt   = torch.load(os.path.join(embed_dir, fname), map_location='cpu', weights_only=False)
        emb  = pt['embeddings'] if isinstance(pt, dict) else pt   # [T, 258]

        # Align lengths
        T = min(len(binary_labels), emb.shape[0])
        emb           = emb[:T]
        binary_labels = binary_labels[:T]

        yield session, location, clip_id, rec_idx, emb, binary_labels


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(teacher: TeacherMLP, embed_dir: str, label_root: str,
             batch_size: int = 1024, device: str = 'cpu'):
    """Run teacher inference on all clips and return per-session metrics."""

    # Accumulate preds/labels keyed by session
    session_preds  = {}
    session_labels = {}

    for session, location, clip_id, rec_idx, emb, labels in iter_clips(embed_dir, label_root):
        # Batch inference
        all_preds = []
        for i in range(0, len(emb), batch_size):
            x = emb[i : i + batch_size].to(device)
            with torch.no_grad():
                logits = teacher(x)           # [N, 2]
                preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
        all_preds = np.concatenate(all_preds)

        key = session
        if key not in session_preds:
            session_preds[key]  = []
            session_labels[key] = []
        session_preds[key].append(all_preds)
        session_labels[key].append(labels)

    # Compute metrics per session
    rows = []
    all_p_combined = []
    all_l_combined = []

    for session in sorted(session_preds.keys()):
        p = np.concatenate(session_preds[session])
        l = np.concatenate(session_labels[session])
        all_p_combined.append(p)
        all_l_combined.append(l)

        f1  = f1_score(l, p, zero_division=0)
        pre = precision_score(l, p, zero_division=0)
        rec = recall_score(l, p, zero_division=0)
        acc = accuracy_score(l, p)
        tn, fp, fn, tp = confusion_matrix(l, p, labels=[0, 1]).ravel()
        ped_ratio = l.mean()

        rows.append({
            'Session':   session,
            'F1':        round(f1,  4),
            'Precision': round(pre, 4),
            'Recall':    round(rec, 4),
            'Accuracy':  round(acc, 4),
            'TP': int(tp), 'FP': int(fp),
            'TN': int(tn), 'FN': int(fn),
            'Ped_ratio': round(float(ped_ratio), 4),
            'N_sec':     len(l),
        })

    # Overall
    p_all = np.concatenate(all_p_combined)
    l_all = np.concatenate(all_l_combined)
    f1_all  = f1_score(l_all, p_all, zero_division=0)
    pre_all = precision_score(l_all, p_all, zero_division=0)
    rec_all = recall_score(l_all, p_all, zero_division=0)
    acc_all = accuracy_score(l_all, p_all)
    tn, fp, fn, tp = confusion_matrix(l_all, p_all, labels=[0, 1]).ravel()

    rows.append({
        'Session':   '--- OVERALL ---',
        'F1':        round(f1_all,  4),
        'Precision': round(pre_all, 4),
        'Recall':    round(rec_all, 4),
        'Accuracy':  round(acc_all, 4),
        'TP': int(tp), 'FP': int(fp),
        'TN': int(tn), 'FN': int(fn),
        'Ped_ratio': round(float(l_all.mean()), 4),
        'N_sec':     len(l_all),
    })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

TEACHER_CKPT = (
    "/media/backup_SSD/Yonghyun/PedestrianDetection_KD/"
    "work_dir/Teacher_existence/Session_6012023/epoch=77-step=8796424.ckpt"
)
EMBED_DIR   = "/media/ykim/Linux/ASPEDv1_VideoEmbeddings"
LABEL_ROOT  = "/media/ykim/Linux/ASPED_v1_npy"


def log_to_wandb(df: pd.DataFrame, args):
    """Log per-session and overall results to a WandB run."""
    if not WANDB_AVAILABLE:
        print("[WandB] wandb not installed — skipping.")
        return

    run = wandb.init(
        project='AcousticPedestrianCounting_KD',
        name='teacher_video_only_eval',
        tags=['teacher', 'video_only', '258dim'],
        config={
            'teacher_ckpt': args.teacher_ckpt,
            'embed_dir':    args.embed_dir,
            'label_root':   args.label_root,
            'input_dim':    258,
            'radius':       '6m',
            'task':         'binary_detection',
        },
    )

    # Overall row
    overall = df[df['Session'] == '--- OVERALL ---'].iloc[0]
    run.summary.update({
        'overall/f1':           overall['F1'],
        'overall/precision':    overall['Precision'],
        'overall/recall':       overall['Recall'],
        'overall/accuracy':     overall['Accuracy'],
        'overall/macro_acc':    (overall['Recall'] + overall['TN'] / (overall['TN'] + overall['FP'])) / 2,
        'overall/tp':           int(overall['TP']),
        'overall/fp':           int(overall['FP']),
        'overall/tn':           int(overall['TN']),
        'overall/fn':           int(overall['FN']),
        'overall/ped_ratio':    overall['Ped_ratio'],
        'overall/n_seconds':    int(overall['N_sec']),
    })

    # Per-session table
    session_rows = df[df['Session'] != '--- OVERALL ---'].copy()
    session_rows['Specificity'] = session_rows['TN'] / (session_rows['TN'] + session_rows['FP'])
    session_rows['MacroAcc']    = (session_rows['Recall'] + session_rows['Specificity']) / 2
    run.log({'per_session_results': wandb.Table(dataframe=session_rows)})

    # Per-session scalars (so they appear as separate bar charts)
    for _, row in session_rows.iterrows():
        run.log({
            f"session/{row['Session']}/f1":        row['F1'],
            f"session/{row['Session']}/recall":    row['Recall'],
            f"session/{row['Session']}/precision": row['Precision'],
            f"session/{row['Session']}/macro_acc": row['MacroAcc'],
        })

    run.finish()
    print(f"[WandB] Results logged to project 'AcousticPedestrianCounting_KD'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_ckpt', default=TEACHER_CKPT)
    parser.add_argument('--embed_dir',    default=EMBED_DIR)
    parser.add_argument('--label_root',   default=LABEL_ROOT)
    parser.add_argument('--device',       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-wandb',     action='store_true')
    args = parser.parse_args()

    print(f"\nDevice: {args.device}")
    print(f"Embed dir: {args.embed_dir}")
    print(f"Label root: {args.label_root}")
    print(f"Teacher ckpt: {args.teacher_ckpt}\n")

    teacher = load_teacher(args.teacher_ckpt, input_dim=258, device=args.device)

    print("Running inference on all clips...")
    df = evaluate(teacher, args.embed_dir, args.label_root, device=args.device)

    # Print
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print("\n" + "="*90)
    print("TEACHER VIDEO-ONLY EVALUATION  (258-dim embeddings, binary detection @ 6m)")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90)

    # Save CSV
    out_csv = os.path.join(os.path.dirname(__file__), 'work_dir', 'teacher_eval.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

    # WandB
    if not args.no_wandb:
        log_to_wandb(df, args)


if __name__ == '__main__':
    main()
