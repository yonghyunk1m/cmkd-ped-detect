"""
train_teacher.py — LOSO training of the video-only Teacher MLP.

Model:  compress_mlp [Linear(258→128) + BN1d + ReLU] + classifier_exist [Linear(128→2)]
Input:  Pre-computed 258-dim Mask2Former embeddings (256 visual + confidence + GT count)
Target: Binary pedestrian detection (0 = no ped, 1 = ped present)
Loss:   Focal Loss (handles ~8% positive-class imbalance)
Logs:   train/loss, val/loss, val/f1, val/recall, val/specificity, val/macro_acc → WandB

Usage:
    # Single fold
    python train_teacher.py --test_session Session_6012023

    # All LOSO folds
    for s in Session_5242023 Session_6012023 Session_6072023 Session_6212023 Session_6282023; do
        python train_teacher.py --test_session $s
    done
"""

import os
import re
import math
import argparse
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

EMBED_DIR  = "/path/to/teacher_embeddings"
LABEL_ROOT = "/path/to/ASPED_v.a"
SAVE_DIR   = "work_dir/teacher_video_only"

SESSIONS = [
    "Session_5242023",
    "Session_6012023",
    "Session_6072023",
    "Session_6212023",
    "Session_6282023",
]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TeacherEmbeddingDataset(Dataset):
    """Each item is one second: (emb_vec [D], binary_label)."""

    def __init__(self, items: list, input_dim: int = 258):
        # items: list of (emb_tensor[T,258], labels[T]) pairs → explode to per-second
        self.embs   = []
        self.labels = []
        for emb, lbls in items:
            T = min(len(emb), len(lbls))
            self.embs.append(emb[:T])
            self.labels.append(lbls[:T])
        self.embs   = torch.cat(self.embs,   dim=0).float()   # [N, 258]
        self.labels = torch.cat(self.labels, dim=0).long()    # [N]
        # Strip metadata dims if using 256-dim visual features only
        if input_dim < self.embs.shape[-1]:
            self.embs = self.embs[:, :input_dim]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embs[idx], self.labels[idx]


_EMB_PATTERN = re.compile(
    r'^(Session_\d+)_(\w+)_(\d+)_6m_ellipse_(\d+)_embed_258\.pt$'
)

def _load_items(embed_dir: str, label_root: str, sessions: list):
    """Return list of (emb_tensor, binary_label_tensor) one entry per clip."""
    items = []
    for fname in sorted(os.listdir(embed_dir)):
        m = _EMB_PATTERN.match(fname)
        if not m:
            continue
        session, location, clip_id, rec_idx = m.groups()
        if session not in sessions:
            continue
        rec_idx = int(rec_idx)

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

        binary = (df[col].values.astype(int) > 0).astype(int)
        pt     = torch.load(os.path.join(embed_dir, fname),
                            map_location='cpu', weights_only=False)
        emb    = pt['embeddings'] if isinstance(pt, dict) else pt  # [T, 258]

        T = min(len(binary), emb.shape[0])
        items.append((emb[:T].float(), torch.tensor(binary[:T], dtype=torch.long)))

    return items


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None
        self.gamma = gamma

    def forward(self, logits, targets):       # [N,2], [N]
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Model
# ─────────────────────────────────────────────────────────────────────────────

class TeacherVideoModel(pl.LightningModule):
    """
    Video-only teacher MLP.

    Architecture (matches pre-trained checkpoint layout):
      compress_mlp    : Linear(input_dim → 128) → BatchNorm1d(128) → ReLU
      classifier_exist: Linear(128 → 2)

    Logs per step/epoch:
      train/loss, val/loss
      val/f1, val/recall (sensitivity), val/specificity, val/macro_acc
      val/tp, val/fp, val/tn, val/fn
    """

    def __init__(
        self,
        input_dim:    int   = 258,
        lr:           float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: list = None,
        focal_gamma:  float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.compress_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.classifier_exist = nn.Linear(128, 2)

        cw = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
        self.criterion = FocalLoss(weight=cw, gamma=focal_gamma)

        self.val_cm   = ConfusionMatrix(task='binary', num_classes=2)
        self.val_f1   = F1Score(task='binary')
        self.val_rec  = Recall(task='binary')
        self.val_prec = Precision(task='binary')

        self.test_cm   = ConfusionMatrix(task='binary', num_classes=2)
        self.test_f1   = F1Score(task='binary')
        self.test_rec  = Recall(task='binary')
        self.test_prec = Precision(task='binary')

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, x):         # x: [N, 258]
        return self.classifier_exist(self.compress_mlp(x))  # [N, 2]

    # ── steps ───────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        emb, labels = batch
        logits = self(emb)
        loss   = self.criterion(logits, labels)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        emb, labels = batch
        logits = self(emb)
        loss   = self.criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        self.val_cm.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.val_rec.update(preds, labels)
        self.val_prec.update(preds, labels)

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        cm = self.val_cm.compute()    # [2,2]: [[TN,FP],[FN,TP]]
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

        recall      = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        macro_acc   = (recall + specificity) / 2.0
        f1          = self.val_f1.compute()
        prec        = self.val_prec.compute()

        self.log_dict({
            'val/f1':          f1,
            'val/precision':   prec,
            'val/recall':      recall,
            'val/specificity': specificity,
            'val/macro_acc':   macro_acc,
            'val/tp':          tp.float(),
            'val/fp':          fp.float(),
            'val/tn':          tn.float(),
            'val/fn':          fn.float(),
        }, prog_bar=False)

        self.val_cm.reset()
        self.val_f1.reset()
        self.val_rec.reset()
        self.val_prec.reset()

    def test_step(self, batch, batch_idx):
        emb, labels = batch
        logits = self(emb)
        loss   = self.criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        self.test_cm.update(preds, labels)
        self.test_f1.update(preds, labels)
        self.test_rec.update(preds, labels)
        self.test_prec.update(preds, labels)

        self.log('test/loss', loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

        recall      = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        macro_acc   = (recall + specificity) / 2.0
        f1          = self.test_f1.compute()
        prec        = self.test_prec.compute()

        self.log_dict({
            'test/f1':          f1,
            'test/precision':   prec,
            'test/recall':      recall,
            'test/specificity': specificity,
            'test/macro_acc':   macro_acc,
            'test/tp':          tp.float(),
            'test/fp':          fp.float(),
            'test/tn':          tn.float(),
            'test/fn':          fn.float(),
        })

        self.test_cm.reset()
        self.test_f1.reset()
        self.test_rec.reset()
        self.test_prec.reset()

    # ── optimizer ───────────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {'scheduler': sched, 'monitor': 'val/loss', 'interval': 'epoch'},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Class weight helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(dataset: TeacherEmbeddingDataset, n_classes: int = 2):
    counts = Counter(dataset.labels.tolist())
    total  = len(dataset)
    raw    = [1.0 / math.sqrt(max(counts.get(i, 1), 1)) for i in range(n_classes)]
    s      = sum(raw)
    normed = [(w / s) * n_classes for w in raw]
    print("\nClass distribution (train only):")
    for i in range(n_classes):
        name = ["No_Ped", "Ped"][i] if n_classes == 2 else f"Class {i}"
        pct  = counts.get(i, 0) / total * 100
        print(f"  {name}: {counts.get(i,0):,} ({pct:.1f}%)  weight={normed[i]:.4f}")
    return normed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_session',  type=str, required=True,
                        choices=SESSIONS, help='LOSO hold-out session')
    parser.add_argument('--embed_dir',     type=str, default=EMBED_DIR)
    parser.add_argument('--label_root',    type=str, default=LABEL_ROOT)
    parser.add_argument('--save_dir',      type=str, default=SAVE_DIR)
    parser.add_argument('--max_epochs',    type=int, default=1000)
    parser.add_argument('--batch_size',    type=int, default=4096)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--focal_gamma',   type=float, default=2.0)
    parser.add_argument('--patience',      type=int, default=15)
    parser.add_argument('--no-wandb',      action='store_true')
    parser.add_argument('--input_dim',    type=int, default=258,
                        help='Embedding dim: 258 (with metadata) or 256 (visual only)')
    args = parser.parse_args()

    pl.seed_everything(42, workers=True)

    ts = time.strftime("%H:%M:%S", time.localtime())
    run_name = f"teacher_video_only_LOSO_{args.test_session}_{ts}"

    # ── Data ────────────────────────────────────────────────────────────────
    train_sessions = [s for s in SESSIONS if s != args.test_session]
    test_sessions  = [args.test_session]

    print(f"\nTrain sessions : {train_sessions}")
    print(f"Test  session  : {test_sessions}\n")

    train_val_items = _load_items(args.embed_dir, args.label_root, train_sessions)
    test_items      = _load_items(args.embed_dir, args.label_root, test_sessions)

    if not train_val_items:
        raise RuntimeError("No training embeddings found. Check embed_dir / label_root paths.")

    full_ds  = TeacherEmbeddingDataset(train_val_items, input_dim=args.input_dim)
    test_ds  = TeacherEmbeddingDataset(test_items, input_dim=args.input_dim)

    # 90/10 train/val split (same ratio as audio model)
    n_val    = max(1, int(0.1 * len(full_ds)))
    n_train  = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Class weights from train set only
    # Build a minimal dataset to count train labels
    train_only_ds = TeacherEmbeddingDataset(
        [(full_ds.embs[i].unsqueeze(0), full_ds.labels[i].unsqueeze(0)) for i in train_ds.indices]
    )
    class_weights = compute_class_weights(train_only_ds)

    # Weighted sampler for train
    sample_weights = torch.tensor(
        [class_weights[l] for l in train_only_ds.labels.tolist()], dtype=torch.float
    )
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    nw = min(4, os.cpu_count())
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=nw, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=nw, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=nw, pin_memory=True)

    # ── Model ───────────────────────────────────────────────────────────────
    model = TeacherVideoModel(
        input_dim=args.input_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
    )

    print(f"\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    # ── Callbacks ───────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(args.save_dir, args.test_session)
    os.makedirs(ckpt_dir, exist_ok=True)

    local_tmp = os.path.join(args.save_dir, '.tmp')
    os.makedirs(local_tmp, exist_ok=True)
    os.environ['TMPDIR'] = os.environ['TMP'] = os.environ['TEMP'] = local_tmp

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stop = EarlyStopping(monitor='val/loss', patience=args.patience,
                                mode='min', verbose=True)

    # ── Logger ──────────────────────────────────────────────────────────────
    if args.no_wandb:
        logger = CSVLogger(save_dir=args.save_dir, name=run_name)
    else:
        logger = WandbLogger(
            project='AcousticPedestrianCounting_KD',
            name=run_name,
            tags=['teacher', 'video_only', f'{args.input_dim}dim', 'LOSO', args.test_session],
            config={
                'model':         'TeacherVideoMLP',
                'architecture':  f'Linear({args.input_dim}→128)+BN+ReLU → Linear(128→2)',
                'input_dim':     args.input_dim,
                'hidden_dim':    128,
                'output_dim':    2,
                'n_params':      total_params,
                'loss':          f'FocalLoss(gamma={args.focal_gamma})',
                'optimizer':     'AdamW + ReduceLROnPlateau(factor=0.5, patience=5)',
                'lr':            args.lr,
                'weight_decay':  args.weight_decay,
                'batch_size':    args.batch_size,
                'max_epochs':    args.max_epochs,
                'patience':      args.patience,
                'test_session':  args.test_session,
                'train_sessions': train_sessions,
                'embed_dir':     args.embed_dir,
                'label_root':    args.label_root,
                'class_weights': class_weights,
                'n_train_sec':   n_train,
                'n_val_sec':     n_val,
                'n_test_sec':    len(test_ds),
            },
        )

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=10,
        callbacks=[ckpt_cb, early_stop],
        logger=logger,
    )

    print(f"{'='*60}")
    print(f" TEACHER VIDEO-ONLY TRAINING  (LOSO: {args.test_session})")
    print(f"{'='*60}")

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    print(f"\n--- Test on {args.test_session} ---")
    trainer.test(model, dataloaders=test_dl, ckpt_path='best')


if __name__ == '__main__':
    main()
