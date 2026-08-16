"""train_kd.py — Knowledge Distillation training entry point.

Usage:
    python train_kd.py \\
        --config configs/kd_binary.yaml \\
        --test_session Session_6012023

    python train_kd.py \\
        --config configs/kd_4class.yaml \\
        --test_session Session_6012023 \\
        --no-wandb

Experiment matrix (change kd.query_agg_type in config):
    Exp A: binary  + gap       → configs/kd_binary.yaml  (query_agg_type: gap)
    Exp B: binary  + attention → configs/kd_binary.yaml  (query_agg_type: attention)
    Exp C: 4class  + gap       → configs/kd_4class.yaml  (query_agg_type: gap)
    Exp D: 4class  + attention → configs/kd_4class.yaml  (query_agg_type: attention)
"""

import os
import argparse
import yaml
import time
import math
from collections import Counter

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import random_split, DataLoader, Subset
import random

from data.utils import get_train_test_splits
from data.dataset import ASPEDDatasetKD, preload_all_embeddings
from data.sampler import create_weighted_sampler
from models.kd_model import ASPEDKDModel


# --------------------------------------------------------------------------- #
# Helpers (mirrors train.py)
# --------------------------------------------------------------------------- #

def verify_class_weights(data_list, n_classes=2):
    print("\n" + "=" * 70)
    print(f"📊 KD DATA DISTRIBUTION & SMOOTHED WEIGHT ANALYSIS")
    print("=" * 70)
    if n_classes == 2:
        labels = [1 if int(l) > 0 else 0 for item in data_list for l in item['label']]
    else:
        labels = [min(int(l), 3) for item in data_list for l in item['label']]
    counts = Counter(labels)
    total  = len(labels)
    raw_weights = [
        1.0 / math.sqrt(counts.get(i, 0)) if counts.get(i, 0) > 0 else 0.0
        for i in range(n_classes)
    ]
    s = sum(raw_weights)
    normalized = [(w / s) * n_classes for w in raw_weights] if s > 0 else [1.0] * n_classes
    for i in range(n_classes):
        name = (["No_Ped", "Ped_Present"] if n_classes == 2
                else [f"Class_{j}" for j in range(n_classes)])[i]
        ratio = counts.get(i, 0) / total * 100 if total > 0 else 0
        print(f"{name:12}: {counts.get(i,0):10,} ({ratio:5.1f}%) | Weight: {normalized[i]:.4f}")
    print("=" * 70 + "\n")
    return normalized


def apply_class_weight_multipliers(class_weights, multipliers):
    if class_weights is None or multipliers is None:
        return class_weights
    if len(multipliers) != len(class_weights):
        raise ValueError(f"Multipliers length mismatch: {len(multipliers)} vs {len(class_weights)}")
    adjusted = [w * float(m) for w, m in zip(class_weights, multipliers)]
    print(f"🔧 Adjusted class weights: {[round(w, 4) for w in adjusted]}")
    return adjusted


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# DataModule shim (no separate class — build loaders inline to reuse sampler)
# --------------------------------------------------------------------------- #

class KDDataModule(pl.LightningDataModule):
    def __init__(self, train_val_list, test_list, batch_size, task_type,
                 num_workers, n_classes, strip_metadata=False):
        super().__init__()
        self.train_val_list  = train_val_list
        self.test_list       = test_list
        self.batch_size      = batch_size
        self.task_type       = task_type
        self.num_workers     = num_workers
        self.n_classes       = n_classes
        self.strip_metadata  = strip_metadata

    def setup(self, stage=None):
        if stage in ('fit', None):
            total      = len(self.train_val_list)
            train_size = int(0.9 * total)
            val_size   = total - train_size
            train_sub, val_sub = random_split(
                self.train_val_list, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            train_items = [self.train_val_list[i] for i in train_sub.indices]
            val_items   = [self.train_val_list[i] for i in val_sub.indices]

            self.train_ds     = ASPEDDatasetKD(train_items, task_type=self.task_type,
                                               max_class=self.n_classes - 1,
                                               strip_metadata=self.strip_metadata)
            self.full_val_ds  = ASPEDDatasetKD(val_items,   task_type=self.task_type,
                                               max_class=self.n_classes - 1,
                                               strip_metadata=self.strip_metadata)

            random.seed(42)
            small_size = min(2000, len(val_items))
            small_idx  = random.sample(range(len(val_items)), small_size)
            self.small_val_ds = Subset(self.full_val_ds, small_idx)

            # Build sampler labels (window-level majority class)
            if self.task_type == 'classification':
                if self.n_classes == 2:
                    self.train_labels = [
                        1 if sum(1 for l in item['label'] if l > 0) >= 5 else 0
                        for item in train_items
                    ]
                else:
                    def _mode(item):
                        ls = [min(int(l), self.n_classes - 1) for l in item['label']]
                        return max(set(ls), key=ls.count)
                    self.train_labels = [_mode(item) for item in train_items]

        if stage in ('test', None):
            self.test_ds = ASPEDDatasetKD(self.test_list, task_type=self.task_type,
                                          max_class=self.n_classes - 1,
                                          strip_metadata=self.strip_metadata)

    def train_dataloader(self):
        kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers,
                      pin_memory=True, persistent_workers=(self.num_workers > 0),
                      drop_last=True)
        if self.task_type == 'classification':
            return DataLoader(self.train_ds, sampler=create_weighted_sampler(self.train_labels), **kwargs)
        return DataLoader(self.train_ds, shuffle=True, **kwargs)

    def val_dataloader(self):
        return DataLoader(self.small_val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=(self.num_workers > 0))

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=(self.num_workers > 0))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    torch.set_float32_matmul_precision('high')
    start_time = time.strftime("%H:%M:%S", time.localtime())

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',        type=str, required=True)
    parser.add_argument('--test_session',  type=str, required=True)
    parser.add_argument('--feature_dir',   type=str, default=None)
    parser.add_argument('--embed_dir',     type=str, default=None)
    parser.add_argument('--teacher_ckpt',  type=str, default=None,
                        help='Override teacher checkpoint. If not set, auto-detected '
                             'from work_dir/teacher_video_only/<test_session>/best-*.ckpt')
    parser.add_argument('--teacher_dir',   type=str, default=None,
                        help='Directory with per-session teacher checkpoints '
                             '(default: work_dir/teacher_video_only)')
    parser.add_argument('--no-wandb',      action='store_true')
    parser.add_argument('--wandb-project', type=str, default='AcousticPedestrianCounting_KD')
    parser.add_argument('--resume', action='store_true', help='Resume training from last.ckpt')
    parser.add_argument('--clip_filter', type=str, default=None,
                        help='Restrict training/test to clips matching this stem (e.g. "0001")')
    parser.add_argument('--require_embedding', action='store_true',
                        help='Skip items without a resolved embedding_path')
    args = parser.parse_args()

    config = load_config(args.config)
    n_classes = config['model'].get('n_classes', 2)
    agg_type  = config['model']['kd'].get('query_agg_type', 'attention')
    cfg_name  = os.path.splitext(os.path.basename(args.config))[0]
    prefix    = f"KD_CLS{n_classes}_{agg_type}"
    run_name  = f"{cfg_name}_LOSO_{args.test_session}_{start_time}"

    pl.seed_everything(config.get('seed', 42), workers=True)

    # ---- Teacher checkpoint (per-fold auto-detection) -------------------- #
    if args.teacher_ckpt:
        teacher_ckpt = args.teacher_ckpt
    else:
        # Look for best-*.ckpt in teacher_dir/{test_session}/
        _teacher_base = (
            args.teacher_dir
            or config.get('model', {}).get('kd', {}).get('teacher_dir', None)
            or os.path.join(os.path.dirname(os.path.abspath(args.config)),
                            '..', 'work_dir', 'teacher_video_only')
        )
        _loso_dir = os.path.join(_teacher_base, args.test_session)
        _loso_dir = os.path.normpath(_loso_dir)
        _candidates = sorted([
            f for f in os.listdir(_loso_dir)
            if f.startswith('best-') and f.endswith('.ckpt')
        ]) if os.path.isdir(_loso_dir) else []
        if _candidates:
            teacher_ckpt = os.path.join(_loso_dir, _candidates[0])
            print(f"[KD] Auto-selected teacher ckpt: {teacher_ckpt}")
        else:
            # No silent fallback. A single shared checkpoint would put the
            # held-out session inside the teacher head's training data for
            # every fold but one, i.e. leakage. Fail loudly instead.
            raise FileNotFoundError(
                f"No per-fold teacher checkpoint found in {_loso_dir}.\n"
                f"Expected <teacher_dir>/{args.test_session}/best-*.ckpt.\n"
                f"Train the LOSO teachers first (train_teacher.py), or pass "
                f"--teacher_ckpt explicitly if you intend to share one teacher "
                f"across folds (this leaks the held-out session and must not be "
                f"used for reported results).")
    config['model']['kd']['teacher_ckpt'] = teacher_ckpt

    # ---- Data ------------------------------------------------------------ #
    feature_dir = (
        args.feature_dir
        or config.get('data_params', {}).get('feature_root_dir')
    )
    embedding_dir = (
        args.embed_dir
        or config.get('data_params', {}).get('embedding_dir')
    )

    # Embedding preload disabled — 11GB preload causes page cache pressure,
    # evicting sdb4 (HDD) pages and triggering heavy HDD re-reads.
    # Embeddings on NVMe are fast enough with lazy loading.
    # if embedding_dir:
    #     preload_all_embeddings(embedding_dir)

    bus_label_dir = config.get('data_params', {}).get('bus_label_dir')
    train_val_list, test_list = get_train_test_splits(
        feature_root_dir=feature_dir,
        test_session=args.test_session,
        window_size_sec=10,
        embedding_dir=embedding_dir,
        bus_label_dir=bus_label_dir,
    )

    # ---- Optional subset filters (for ablation experiments) -------------- #
    if args.clip_filter:
        def _matches_clip(item):
            return f"_{args.clip_filter}.npy" in item['npy_path'] or \
                   f"/{args.clip_filter}.npy" in item['npy_path']
        orig_train, orig_test = len(train_val_list), len(test_list)
        train_val_list = [x for x in train_val_list if _matches_clip(x)]
        test_list      = [x for x in test_list      if _matches_clip(x)]
        print(f"[clip_filter={args.clip_filter}] train: {orig_train} -> {len(train_val_list)}  "
              f"test: {orig_test} -> {len(test_list)}")

    if args.require_embedding:
        orig_train, orig_test = len(train_val_list), len(test_list)
        train_val_list = [x for x in train_val_list if x.get('embedding_path')]
        test_list      = [x for x in test_list      if x.get('embedding_path')]
        print(f"[require_embedding] train: {orig_train} -> {len(train_val_list)}  "
              f"test: {orig_test} -> {len(test_list)}")

    # ---- Class weights --------------------------------------------------- #
    train_size  = int(0.9 * len(train_val_list))
    val_size    = len(train_val_list) - train_size
    pure_train, _ = random_split(
        train_val_list, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    pure_train_list = [train_val_list[i] for i in pure_train.indices]

    class_weights = None
    if config['model']['task'] == 'classification':
        class_weights = verify_class_weights(pure_train_list, n_classes=n_classes)
        multipliers   = config['model'].get('class_weight_multipliers')
        class_weights = apply_class_weight_multipliers(class_weights, multipliers)

    # ---- DataModule ------------------------------------------------------ #
    strip_metadata = config.get('data_params', {}).get('strip_metadata', False)
    dm = KDDataModule(
        train_val_list=train_val_list,
        test_list=test_list,
        batch_size=config['dataloader_params']['batch_size'],
        task_type=config['model']['task'],
        num_workers=config['dataloader_params']['num_workers'],
        n_classes=n_classes,
        strip_metadata=strip_metadata,
    )

    # ---- Model ----------------------------------------------------------- #
    backbone_name = config['model'].get('backbone', {}).get('name', 'vggish')
    backbone_finetune = config['model'].get('backbone', {}).get('finetune', False)
    model = ASPEDKDModel(
        kd_cfg=config['model']['kd'],
        # student kwargs (forwarded to ASPEDLightningModel.__init__)
        token_dim=config['model']['token_dim'],
        nhead=config['model']['nhead'],
        dropout=config['model']['dropout'],
        num_classes=n_classes,
        num_layers=config['model']['num_layers'],
        task_type=config['model']['task'],
        learning_rate=config['model']['optim']['args']['lr'],
        weight_decay=config['model']['optim']['args']['weight_decay'],
        class_weights=class_weights,
        backbone_name=backbone_name,
        backbone_finetune=backbone_finetune,
    )

    # ---- Logging --------------------------------------------------------- #
    save_dir = config['trainer']['logger']['save_dir']
    session_dir = os.path.join(save_dir, args.test_session)
    os.makedirs(session_dir, exist_ok=True)

    local_tmp = os.path.join(save_dir, '.tmp')
    os.makedirs(local_tmp, exist_ok=True)
    os.environ['TMPDIR'] = os.environ['TMP'] = os.environ['TEMP'] = local_tmp

    if args.no_wandb:
        logger = CSVLogger(save_dir=save_dir, name=run_name)
    else:
        wandb_cfg = {
            **config,                          # full yaml config
            'test_session':  args.test_session,
            'feature_dir':   feature_dir,
            'embedding_dir': embedding_dir,
            'n_train_windows': len(train_val_list),
            'n_test_windows':  len(test_list),
            'class_weights':   class_weights,
        }
        logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            save_dir=save_dir,
            config=wandb_cfg,
            tags=[
                cfg_name,
                f"cls{n_classes}",
                agg_type,
                args.test_session,
                'kd',
            ],
        )

    # ---- Callbacks ------------------------------------------------------- #
    es_cfg   = config['trainer']['early_stopping']
    ck_cfg   = config['trainer']['checkpoint']
    monitor  = es_cfg.get('monitor', 'val_quick/loss')
    patience = es_cfg['patience']
    es_mode  = es_cfg.get('mode', 'min')

    ck_monitor  = ck_cfg.get('monitor', monitor)
    ck_mode     = ck_cfg.get('mode', es_mode)
    ck_filename = 'best-{epoch:02d}'

    checkpoint_cb = ModelCheckpoint(
        dirpath=session_dir,
        filename=ck_filename,
        save_top_k=ck_cfg.get('save_top_k', 1),
        monitor=ck_monitor,
        mode=ck_mode,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor=monitor, patience=patience, mode=es_mode, verbose=True
    )

    # ---- Trainer --------------------------------------------------------- #
    accumulate = config['trainer']['args'].get('accumulate_grad_batches', 1)
    trainer = pl.Trainer(
        max_epochs=config['trainer']['args']['max_epochs'],
        accelerator=config['trainer']['args']['accelerator'],
        devices=config['trainer']['args']['devices'],
        val_check_interval=config['trainer']['args'].get('val_check_interval', 1.0),
        precision=config['trainer']['args']['precision'],
        accumulate_grad_batches=accumulate,
        log_every_n_steps=1,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
    )

    # ---- Fit & Test ------------------------------------------------------ #
    ckpt_path = None
    if args.resume:
        last_ckpt = os.path.join(checkpoint_cb.dirpath, "last.ckpt")
        if os.path.isfile(last_ckpt):
            ckpt_path = last_ckpt
            print(f"▶ Resuming from {last_ckpt}")
        else:
            print(f"⚠ --resume set but {last_ckpt} not found. Starting from scratch.")
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    print(f"🎯 KD Training complete. Running test on: {args.test_session}")
    trainer.test(model, datamodule=dm, ckpt_path='best')


if __name__ == '__main__':
    main()
