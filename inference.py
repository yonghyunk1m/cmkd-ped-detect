"""
inference.py — Run a trained checkpoint on all audio data and save per-second predictions.

Outputs a CSV per (session, location, recorder, clip) with columns:
    second, prediction, [prob_class_0, prob_class_1, ...]

Usage:
    # 4-class on all v3 data
    python inference.py \
        --checkpoint work_dir/baseline_4class_v3/Session_02152024/best-epoch=03-val_quick/macro_accuracy=0.4208.ckpt \
        --config configs/baseline_4class_v3.yaml \
        --data_dir /media/backup_SSD/ASPED_v3_npy \
        --output_dir results/predictions_4class_v3

    # Binary on all v1 data
    python inference.py \
        --checkpoint work_dir/kd_binary/Session_5242023/best-*.ckpt \
        --config configs/kd_binary.yaml \
        --data_dir /media/ykim/Linux/ASPED_v1_npy \
        --output_dir results/predictions_kd_binary_v1 \
        --is_kd

    # Run all LOSO folds at once (uses best checkpoint per fold)
    python inference.py \
        --checkpoint_dir work_dir/baseline_4class_v3 \
        --config configs/baseline_4class_v3.yaml \
        --data_dir /media/backup_SSD/ASPED_v3_npy \
        --output_dir results/predictions_4class_v3
"""

import os
import glob
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))

from models.seq2seq import ASPEDLightningModel
from models.kd_model import ASPEDKDModel


def load_model(checkpoint_path, config, is_kd=False, device='cuda'):
    """Load a trained model from checkpoint."""
    n_classes = config['model']['n_classes']
    token_dim = config['model'].get('token_dim', 128)
    nhead = config['model'].get('nhead', 4)
    dropout = config['model'].get('dropout', 0.2)
    num_layers = config['model'].get('num_layers', 1)
    task_type = config['model'].get('task', 'classification')

    if is_kd:
        kd_cfg = config['model']['kd']
        model = ASPEDKDModel(
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

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device)
    model.eval()
    return model, n_classes


def find_best_checkpoint(session_dir):
    """Find the best checkpoint in a session directory."""
    pattern = os.path.join(session_dir, 'best-*', '*.ckpt')
    matches = glob.glob(pattern)
    if not matches:
        # Try flat structure
        matches = glob.glob(os.path.join(session_dir, 'best-*.ckpt'))
    return matches[0] if matches else None


def scan_audio_files(data_dir):
    """Scan data directory and return list of (session, location, recorder, clip_stem, npy_path, csv_path)."""
    items = []
    sessions = sorted([d for d in os.listdir(data_dir) if d.startswith('Session_')])

    for session in sessions:
        session_path = os.path.join(data_dir, session)
        locations = sorted([d for d in os.listdir(session_path)
                           if os.path.isdir(os.path.join(session_path, d))])

        for location in locations:
            audio_dir = os.path.join(session_path, location, 'Audio')
            labels_dir = os.path.join(session_path, location, 'Labels')
            if not os.path.exists(audio_dir):
                continue

            recorders = sorted([d for d in os.listdir(audio_dir)
                               if os.path.isdir(os.path.join(audio_dir, d))])

            # Get label files for reference (to know clip stems)
            label_files = []
            if os.path.exists(labels_dir):
                label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.csv')])

            for rec_idx, recorder in enumerate(recorders):
                rec_dir = os.path.join(audio_dir, recorder)
                npy_files = sorted([f for f in os.listdir(rec_dir) if f.endswith('.npy')])

                for npy_file in npy_files:
                    clip_stem = npy_file.replace('.npy', '')
                    npy_path = os.path.join(rec_dir, npy_file)
                    csv_path = os.path.join(labels_dir, clip_stem + '.csv') if os.path.exists(labels_dir) else None
                    if csv_path and not os.path.exists(csv_path):
                        csv_path = None

                    items.append({
                        'session': session,
                        'location': location,
                        'recorder': recorder,
                        'rec_idx': rec_idx + 1,
                        'clip_stem': clip_stem,
                        'npy_path': npy_path,
                        'csv_path': csv_path,
                    })

    return items


@torch.no_grad()
def run_inference(model, npy_path, n_classes, device='cuda', window_sec=10, sr=16000, stride_sec=10):
    """Run sliding-window inference on a single NPY file.

    Returns:
        predictions: np.array [T] — predicted class per second
        probabilities: np.array [T, n_classes] — class probabilities per second
    """
    try:
        audio = np.load(npy_path, mmap_mode='r')
    except Exception:
        return None, None

    total_sec = audio.shape[0] // sr
    if total_sec < window_sec:
        return None, None

    # Accumulate predictions per second (voting from overlapping windows)
    prob_sum = np.zeros((total_sec, n_classes if n_classes > 2 else 2), dtype=np.float64)
    count = np.zeros(total_sec, dtype=np.int32)

    # Stride controls overlap: window_sec=non-overlapping, 1=fully overlapping (avg)
    stride = stride_sec
    for start in range(0, total_sec - window_sec + 1, stride):
        chunk = audio[start * sr: (start + window_sec) * sr]
        x = torch.from_numpy(chunk.copy()).float().unsqueeze(0).to(device)  # [1, 160000]
        logits = model(x)  # [1, 10, n_classes] or [1, 10, 1]

        if n_classes == 2:
            # Binary: logits shape [1, 10, 1]
            logits_2d = logits.squeeze(0).squeeze(-1)  # [10]
            probs = torch.stack([1 - torch.sigmoid(logits_2d), torch.sigmoid(logits_2d)], dim=-1)  # [10, 2]
        else:
            # Multi-class: logits shape [1, 10, n_classes]
            probs = F.softmax(logits.squeeze(0), dim=-1)  # [10, n_classes]

        probs_np = probs.cpu().numpy()
        for t in range(window_sec):
            sec = start + t
            if sec < total_sec:
                prob_sum[sec] += probs_np[t]
                count[sec] += 1

    # Average probabilities
    valid = count > 0
    prob_avg = np.zeros_like(prob_sum)
    prob_avg[valid] = prob_sum[valid] / count[valid, np.newaxis]

    predictions = prob_avg.argmax(axis=1)

    return predictions, prob_avg


def main():
    parser = argparse.ArgumentParser(description='Run inference on all audio data')
    parser.add_argument('--checkpoint', type=str, default=None, help='Single checkpoint path')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory with per-session checkpoints (uses best per fold)')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory of NPY data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for predictions')
    parser.add_argument('--is_kd', action='store_true', help='Model is KD (ASPEDKDModel)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--stride', type=int, default=10,
                        help='Sliding window stride in seconds (1=fully overlapping, 10=non-overlapping)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip clips that already have output CSVs (for crash recovery)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Scan all audio files
    items = scan_audio_files(args.data_dir)
    print(f"Found {len(items)} audio clips in {args.data_dir}")

    # Determine checkpoint(s)
    if args.checkpoint:
        # Single checkpoint mode
        print(f"Loading checkpoint: {args.checkpoint}")
        model, n_classes = load_model(args.checkpoint, config, args.is_kd, args.device)
        checkpoints = {None: model}  # None key = use for all sessions
    elif args.checkpoint_dir:
        # Per-session checkpoint mode (LOSO)
        checkpoints = {}
        for session_dir in sorted(glob.glob(os.path.join(args.checkpoint_dir, 'Session_*'))):
            session = os.path.basename(session_dir)
            ckpt_path = find_best_checkpoint(session_dir)
            if ckpt_path:
                print(f"  {session}: {os.path.basename(ckpt_path)}")
                model, n_classes = load_model(ckpt_path, config, args.is_kd, args.device)
                checkpoints[session] = model
            else:
                print(f"  {session}: NO checkpoint found, skipping")
        if not checkpoints:
            print("ERROR: No checkpoints found!")
            return
    else:
        print("ERROR: Provide --checkpoint or --checkpoint_dir")
        return

    # Run inference
    results_summary = []
    skipped = 0
    for item in tqdm(items, desc="Inference"):
        session = item['session']

        # Skip existing output if --skip-existing
        if args.skip_existing:
            out_check = os.path.join(
                args.output_dir, session, item['location'],
                item['recorder'], f"{item['clip_stem']}_predictions.csv")
            if os.path.exists(out_check):
                skipped += 1
                continue

        # Select model: per-session if available, else single model
        if session in checkpoints:
            model = checkpoints[session]
        elif None in checkpoints:
            model = checkpoints[None]
        else:
            continue  # No model for this session

        predictions, probabilities = run_inference(
            model, item['npy_path'], n_classes, args.device,
            stride_sec=args.stride,
        )
        if predictions is None:
            continue

        # Load GT labels if available
        gt_col = f"recorder{item['rec_idx']}_6m"
        gt_labels = None
        if item['csv_path']:
            try:
                df = pd.read_csv(item['csv_path'])
                if gt_col in df.columns:
                    gt_raw = df[gt_col].values[:len(predictions)]
                    gt_labels = np.clip(gt_raw, 0, n_classes - 1)
            except Exception:
                pass

        # Build output dataframe
        out_df = pd.DataFrame({'second': np.arange(len(predictions)), 'prediction': predictions})
        for c in range(probabilities.shape[1]):
            out_df[f'prob_class_{c}'] = probabilities[:len(predictions), c]
        if gt_labels is not None:
            out_df['gt_label'] = gt_labels[:len(predictions)]

        # Save
        out_subdir = os.path.join(args.output_dir, session, item['location'], item['recorder'])
        os.makedirs(out_subdir, exist_ok=True)
        out_path = os.path.join(out_subdir, f"{item['clip_stem']}_predictions.csv")
        out_df.to_csv(out_path, index=False)

        # Summary stats
        results_summary.append({
            'session': session,
            'location': item['location'],
            'recorder': item['recorder'],
            'clip': item['clip_stem'],
            'n_seconds': len(predictions),
            'pred_mean': predictions.mean(),
            'gt_available': gt_labels is not None,
        })

    # Save summary
    summary_path = os.path.join(args.output_dir, 'inference_summary.csv')
    pd.DataFrame(results_summary).to_csv(summary_path, index=False)
    print(f"\nDone! {len(results_summary)} clips processed, {skipped} skipped (existing).")
    print(f"Predictions saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
