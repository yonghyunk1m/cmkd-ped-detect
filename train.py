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
from torch.utils.data import random_split

from data.datamodule import ASPEDDataModule
from data.utils import get_train_test_splits
from models.seq2seq import ASPEDLightningModel

def verify_class_weights(data_list, n_classes=2):
    """
    Analyzes training data distribution and calculates 'Global Smoothed Weights'
    using the 1/sqrt(count) method to prevent over-correction.
    """
    print("\n" + "="*70)
    print(f"📊 GLOBAL DATA DISTRIBUTION & SMOOTHED WEIGHT ANALYSIS")
    print("="*70)
    
    # Extract labels from the provided data list (should be pure training set)
    if n_classes == 2:
        labels = [1 if int(l) > 0 else 0 for item in data_list for l in item['label']]
    else:
        labels = [min(int(l), 3) for item in data_list for l in item['label']]
    
    counts = Counter(labels)
    total = len(labels)
    
    # 1. Calculate Inverse Square Root (Smoothing)
    raw_weights = []
    for i in range(n_classes):
        count = counts.get(i, 0)
        if count > 0:
            raw_weights.append(1.0 / math.sqrt(count))
        else:
            raw_weights.append(0.0)
            
    # 2. Normalize weights so that the sum equals n_classes (keeps loss scale stable)
    sum_weights = sum(raw_weights)
    if sum_weights > 0:
        normalized_weights = [(w / sum_weights) * n_classes for w in raw_weights]
    else:
        normalized_weights = [1.0] * n_classes
    
    # Print analysis results
    for i in range(n_classes):
        count = counts.get(i, 0)
        class_name = "No_Ped" if i == 0 and n_classes == 2 else ("Ped_Present" if i == 1 and n_classes == 2 else f"Class {i}")
        ratio = (count / total) * 100 if total > 0 else 0
        print(f"{class_name:12}: {count:10,} frames ({ratio:5.1f}%) | Weight: {normalized_weights[i]:.4f}")

    print("="*70 + "\n")
    return normalized_weights


def apply_class_weight_multipliers(class_weights, multipliers):
    if class_weights is None or multipliers is None:
        return class_weights
    if len(multipliers) != len(class_weights):
        raise ValueError(
            f"class_weight_multipliers length ({len(multipliers)}) "
            f"must match n_classes ({len(class_weights)})"
        )
    adjusted = [w * float(m) for w, m in zip(class_weights, multipliers)]
    print(f"🔧 Applied class_weight_multipliers: {multipliers}")
    print(f"   -> Adjusted class weights: {[round(w, 4) for w in adjusted]}")
    return adjusted

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    torch.set_float32_matmul_precision('high')
    start_time = time.strftime("%H:%M:%S", time.localtime())
    
    parser = argparse.ArgumentParser(description="Train the ASPED Model (LOSO CV)")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--test_session', type=str, required=True, help='Session to hold out for testing')
    parser.add_argument('--feature_dir', type=str, default=None, help='Override feature root directory')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging (use CSV logger)')
    parser.add_argument('--wandb-project', type=str, default='AcousticPedestrianCounting', help='WandB project name')
    parser.add_argument('--resume', action='store_true', help='Resume training from last.ckpt')
    args = parser.parse_args()

    config = load_config(args.config)
    
    n_classes = config['model'].get('n_classes', 1)
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    prefix = f"CLS{n_classes}" if config['model']['task'] == 'classification' else "REG"
    run_name = f"{cfg_name}_LOSO_{args.test_session}_{start_time}"
    
    pl.seed_everything(config.get('seed', 42), workers=True)

    # 1. Load Data Splits (Test session is already excluded from train_val_list)
    feature_dir = (
        args.feature_dir
        or config.get('data_params', {}).get('feature_root_dir')
        or "/path/to/ASPED_v.c"
    )
    bus_label_dir = config.get('data_params', {}).get('bus_label_dir')
    train_val_list, test_list = get_train_test_splits(
        feature_root_dir=feature_dir, test_session=args.test_session, window_size_sec=10,
        bus_label_dir=bus_label_dir,
    )
    
    # 2. Strict Isolation: Isolate pure training data for weight calculation (excluding validation)
    # Use the same split logic as ASPEDDataModule (90/10, seeded random split).
    train_size = int(0.9 * len(train_val_list))
    val_size = len(train_val_list) - train_size
    pure_train_subset, _ = random_split(
        train_val_list,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    pure_train_list = [train_val_list[i] for i in pure_train_subset.indices]

    # 3. Calculate Weights based ONLY on pure training data
    class_weights = None
    if config['model']['task'] == 'classification':
        class_weights = verify_class_weights(pure_train_list, n_classes=n_classes)
        multipliers = config['model'].get('class_weight_multipliers')
        class_weights = apply_class_weight_multipliers(class_weights, multipliers)
        
    # 4. Initialize DataModule
    datamodule = ASPEDDataModule(
        train_val_data_list=train_val_list,
        test_data_list=test_list,
        batch_size=config['dataloader_params']['batch_size'],
        task_type=config['model']['task'],
        num_workers=config['dataloader_params']['num_workers'],
        n_classes=n_classes
    )

    # 5. Initialize Model with Global Weights
    backbone_name = config['model'].get('backbone', {}).get('name', 'vggish')
    backbone_finetune = config['model'].get('backbone', {}).get('finetune', False)
    model = ASPEDLightningModel(
        token_dim=config['model']['token_dim'],
        nhead=config['model']['nhead'],
        dropout=config['model']['dropout'],
        num_classes=n_classes,
        task_type=config['model']['task'],
        learning_rate=config['model']['optim']['args']['lr'],
        weight_decay=config['model']['optim']['args']['weight_decay'],
        class_weights=class_weights,
        binary_threshold=config['model'].get('binary_threshold', 0.5),
        backbone_name=backbone_name,
        backbone_finetune=backbone_finetune,
    )

    save_dir = config['trainer']['logger']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, args.test_session), exist_ok=True)

    # Keep temporary checkpoint writes on the same filesystem as final ckpt path.
    local_tmp_dir = os.path.join(save_dir, ".tmp")
    os.makedirs(local_tmp_dir, exist_ok=True)
    os.environ["TMPDIR"] = local_tmp_dir
    os.environ["TMP"] = local_tmp_dir
    os.environ["TEMP"] = local_tmp_dir

    es_cfg = config['trainer']['early_stopping']
    ck_cfg = config['trainer'].get('checkpoint', {})
    monitor_metric = es_cfg.get('monitor', 'val_quick/loss')
    patience = es_cfg['patience']
    es_mode = es_cfg.get('mode', 'min')

    ck_monitor = ck_cfg.get('monitor', monitor_metric)
    ck_mode = ck_cfg.get('mode', es_mode)
    ck_filename = 'best-{epoch:02d}'

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, args.test_session),
        filename=ck_filename,
        save_top_k=ck_cfg.get('save_top_k', 1),
        monitor=ck_monitor,
        mode=ck_mode,
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        mode=es_mode,
        verbose=True
    )

    if getattr(args, 'no_wandb', False):
        logger = CSVLogger(save_dir=save_dir, name=run_name)
    else:
        logger = WandbLogger(project=args.wandb_project, name=run_name, save_dir=save_dir)

    accumulate = config['trainer']['args'].get('accumulate_grad_batches', 1)

    trainer = pl.Trainer(
        max_epochs=config['trainer']['args']['max_epochs'],
        accelerator=config['trainer']['args']['accelerator'],
        devices=config['trainer']['args']['devices'],
        val_check_interval=config['trainer']['args'].get('val_check_interval', 1.0),
        precision=config['trainer']['args']['precision'],
        accumulate_grad_batches=accumulate,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger
    )

    # 6. Fit & Test
    ckpt_path = None
    if args.resume:
        last_ckpt = os.path.join(checkpoint_callback.dirpath, "last.ckpt")
        if os.path.isfile(last_ckpt):
            ckpt_path = last_ckpt
            print(f"▶ Resuming from {last_ckpt}")
        else:
            print(f"⚠ --resume set but {last_ckpt} not found. Starting from scratch.")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    
    print(f"🎯 Training complete. Running Full Evaluation on Test Session: {args.test_session}...")
    trainer.test(model, datamodule=datamodule, ckpt_path='best')

if __name__ == "__main__":
    main()