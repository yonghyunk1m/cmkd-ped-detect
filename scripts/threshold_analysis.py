"""
Threshold Analysis: Find optimal binary decision threshold on validation set,
then evaluate on test set. Compares KD vs Baseline fairly.

Usage:
    python scripts/threshold_analysis.py \
        --kd_ckpt   work_dir/kd_binary/Session_5242023/best-*.ckpt \
        --bl_ckpt   work_dir/baseline_binary_v1/Session_5242023/best-*.ckpt \
        --config_kd configs/kd_binary.yaml \
        --config_bl configs/baseline_binary_v1.yaml \
        --test_session Session_5242023
"""
import argparse, glob, os, sys, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.utils import get_train_test_splits
from data.datamodule import ASPEDDataModule
from models.seq2seq import ASPEDLightningModel
from models.kd_model import ASPEDKDModel as ASPEDKDLightningModel


def collect_probs(model, dataloader, device):
    """Run inference, return (probs, labels) as numpy arrays."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:  # KD dataset: (audio, embed, label)
                features, _, labels = batch
            else:
                features, labels = batch
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits.squeeze(-1).reshape(-1)).cpu().numpy()
            labs = (labels > 0).float().reshape(-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labs)
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(probs, labels, threshold):
    preds = (probs > threshold).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    acc_0 = tn / (tn + fp + 1e-8)
    acc_1 = tp / (tp + fn + 1e-8)
    macro_acc = (acc_0 + acc_1) / 2
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        'threshold': threshold,
        'macro_acc': macro_acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'acc_0': acc_0, 'acc_1': acc_1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def find_best_threshold(probs, labels, metric='macro_acc'):
    best, best_t = -1, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        m = compute_metrics(probs, labels, t)
        if m[metric] > best:
            best = m[metric]
            best_t = t
    return best_t


def load_model(ckpt_path, config, is_kd=False):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    n_classes = cfg['model']['n_classes']
    token_dim = cfg['model']['token_dim']
    nhead = cfg['model']['nhead']
    dropout = cfg['model']['dropout']
    num_layers = cfg['model']['num_layers']
    task_type = cfg['model'].get('task', 'classification')

    if is_kd:
        kd_cfg = cfg['model']['kd']
        model = ASPEDKDLightningModel(
            kd_cfg=kd_cfg,
            token_dim=token_dim, nhead=nhead, dropout=dropout,
            num_classes=n_classes, num_layers=num_layers,
            task_type=task_type,
        )
    else:
        model = ASPEDLightningModel(
            token_dim=token_dim, nhead=nhead, dropout=dropout,
            num_classes=n_classes, num_layers=num_layers,
            task_type=task_type,
        )
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    return model


def print_comparison(name, val_default, val_opt, test_default, test_opt, opt_t):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Optimal threshold (on val): {opt_t:.2f}")
    print(f"")
    print(f"  {'':20s} {'Val (t=0.5)':>12s} {'Val (t=opt)':>12s} {'Test (t=0.5)':>12s} {'Test (t=opt)':>12s}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for key in ['macro_acc', 'f1', 'precision', 'recall', 'acc_0', 'acc_1']:
        v_d = val_default[key]
        v_o = val_opt[key]
        t_d = test_default[key]
        t_o = test_opt[key]
        print(f"  {key:20s} {v_d:12.4f} {v_o:12.4f} {t_d:12.4f} {t_o:12.4f}")
    print(f"  {'':20s} {'':>12s} {'':>12s} {'':>12s} {'':>12s}")
    for key in ['tp', 'tn', 'fp', 'fn']:
        print(f"  {key:20s} {int(test_default[key]):>12,d} {int(test_opt[key]):>12,d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kd_ckpt', type=str, required=True)
    parser.add_argument('--bl_ckpt', type=str, required=True)
    parser.add_argument('--config_kd', type=str, default='configs/kd_binary.yaml')
    parser.add_argument('--config_bl', type=str, default='configs/baseline_binary_v1.yaml')
    parser.add_argument('--test_session', type=str, required=True)
    parser.add_argument('--optimize_for', type=str, default='macro_acc',
                        choices=['macro_acc', 'f1'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load configs ---
    with open(args.config_kd) as f:
        kd_cfg = yaml.safe_load(f)
    with open(args.config_bl) as f:
        bl_cfg = yaml.safe_load(f)

    # --- Prepare data (shared for both models) ---
    feature_dir = kd_cfg['data_params']['feature_root_dir']
    embedding_dir = kd_cfg['data_params'].get('embedding_dir')

    train_val_list, test_list = get_train_test_splits(
        feature_root_dir=feature_dir,
        test_session=args.test_session,
        embedding_dir=embedding_dir,
    )

    dm = ASPEDDataModule(
        train_val_data_list=train_val_list,
        test_data_list=test_list,
        batch_size=128,
        num_workers=4,
        n_classes=2,
    )
    dm.setup()

    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    results = {}

    for name, ckpt_path, config_path, is_kd in [
        ('KD', args.kd_ckpt, args.config_kd, True),
        ('Baseline', args.bl_ckpt, args.config_bl, False),
    ]:
        print(f"\n>>> Loading {name}: {ckpt_path}")
        model = load_model(ckpt_path, config_path, is_kd=is_kd).to(device)

        print(f"    Collecting val predictions...")
        val_probs, val_labels = collect_probs(model, val_loader, device)
        print(f"    Collecting test predictions...")
        test_probs, test_labels = collect_probs(model, test_loader, device)

        opt_t = find_best_threshold(val_probs, val_labels, metric=args.optimize_for)

        val_default = compute_metrics(val_probs, val_labels, 0.5)
        val_opt = compute_metrics(val_probs, val_labels, opt_t)
        test_default = compute_metrics(test_probs, test_labels, 0.5)
        test_opt = compute_metrics(test_probs, test_labels, opt_t)

        print_comparison(name, val_default, val_opt, test_default, test_opt, opt_t)
        results[name] = {
            'opt_threshold': opt_t,
            'val_default': val_default,
            'val_opt': val_opt,
            'test_default': test_default,
            'test_opt': test_opt,
        }

    # --- Final side-by-side ---
    print(f"\n{'='*60}")
    print(f"  SIDE-BY-SIDE (test, optimized threshold)")
    print(f"{'='*60}")
    print(f"  {'':20s} {'KD':>12s} {'Baseline':>12s}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    print(f"  {'threshold':20s} {results['KD']['opt_threshold']:12.2f} {results['Baseline']['opt_threshold']:12.2f}")
    for key in ['macro_acc', 'f1', 'precision', 'recall', 'acc_0', 'acc_1']:
        kd_v = results['KD']['test_opt'][key]
        bl_v = results['Baseline']['test_opt'][key]
        winner = ' <--' if kd_v > bl_v else ''
        print(f"  {key:20s} {kd_v:12.4f} {bl_v:12.4f}{winner}")


if __name__ == '__main__':
    main()
