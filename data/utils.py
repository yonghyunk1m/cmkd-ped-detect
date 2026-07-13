import os
import re
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm


def _resolve_embedding_path(embedding_dir: str, session: str, location: str,
                              npy_stem: str, recorder_idx: int) -> Optional[str]:
    """Build the expected .pt path for a given recorder clip.

    Naming convention (from PedestrianDetection_KD/embeddings_dataset/):
        {session}_{location}_{npy_stem}_{recorder_idx}.pt
        e.g. Session_5242023_Cadell_0001_6m_ellipse_1.pt

    Returns the path if the file exists, else None.
    """
    # Try both "stem_6m_ellipse_idx" (ASPEDv1 naming) and bare "stem_idx"
    for base in (
        f"{session}_{location}_{npy_stem}_6m_ellipse_{recorder_idx}",
        f"{session}_{location}_{npy_stem}_{recorder_idx}",
    ):
        for fname in (f"{base}_embed_258.pt", f"{base}.pt"):
            path = os.path.join(embedding_dir, fname)
            if os.path.exists(path):
                return path
    return None


def _build_bus_mask(df, window_size_sec):
    """Build a boolean array marking seconds that fall inside a bus-free window.

    For each candidate start position, the entire window [start, start+window)
    must be bus-free.  Returns a list of valid start seconds.
    """
    if 'busFrame' not in df.columns:
        return None  # no filtering needed
    bus = df['busFrame'].values.astype(int)
    n_bus = int(bus.sum())
    if n_bus == 0:
        return None  # all frames are valid
    # valid[i] = True iff *every* second in [i, i+window) has busFrame == 0
    valid_starts = []
    last = len(bus) - window_size_sec
    for s in range(last + 1):
        if bus[s:s + window_size_sec].sum() == 0:
            valid_starts.append(s)
    return set(valid_starts)


def get_train_test_splits(feature_root_dir, test_session, window_size_sec=10,
                          train_stride_sec=1, test_stride_sec=10,
                          embedding_dir=None, bus_label_dir=None):
    """
    Optimized directory scanner.
    - train_stride_sec=1: 90% overlap for maximized training data.
    - test_stride_sec=10: No overlap for fast, clean baseline evaluation. (Or 1 for sliding window ensemble)
    - bus_label_dir: Optional path to bus-aware labels (ASPED v2).
      When set, CSVs from this directory (which contain a 'busFrame' column)
      are used instead of the default Labels/ directory for matching sessions.
      Windows that overlap any busFrame==1 second are excluded.
    """
    print(f"🚀 Scanning root directory: {feature_root_dir}...")
    if embedding_dir:
        print(f"   Embedding dir : {embedding_dir}")
    if bus_label_dir:
        print(f"   Bus-label dir : {bus_label_dir}")
    train_list = []
    test_list = []
    bus_filtered_total = 0

    sessions = [d for d in os.listdir(feature_root_dir) if d.startswith('Session_')]

    for session in sessions:
        session_path = os.path.join(feature_root_dir, session)
        locations = [d for d in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, d))]

        for location in locations:
            audio_dir  = os.path.join(session_path, location, 'Audio')
            labels_dir = os.path.join(session_path, location, 'Labels')

            # Use bus-aware labels if available for this session/location
            if bus_label_dir:
                bus_csv_dir = os.path.join(bus_label_dir, session, location)
                if os.path.isdir(bus_csv_dir):
                    labels_dir = bus_csv_dir

            if not os.path.exists(audio_dir) or not os.path.exists(labels_dir):
                continue

            recorders = sorted([
                d for d in os.listdir(audio_dir)
                if os.path.isdir(os.path.join(audio_dir, d))
            ])

            col_mapping = {
                rec: f"recorder{i+1}_6m" for i, rec in enumerate(recorders)
            }

            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.csv')]

            for label_file in tqdm(label_files, desc=f"Processing {session} - {location}", leave=False):
                csv_path = os.path.join(labels_dir, label_file)
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"⚠️ [WARNING] Failed to read {csv_path}: {e}")
                    continue

                npy_filename   = label_file.replace('.csv', '.npy')
                npy_stem       = label_file.replace('.csv', '')   # e.g. 0001_6m_ellipse
                last_start_sec = len(df) - window_size_sec

                # Keep exact-length clips (len == window_size_sec) as one valid window.
                if last_start_sec < 0:
                    continue

                # Build bus-frame mask (None if column absent or all-zero)
                valid_starts = _build_bus_mask(df, window_size_sec)

                for i, recorder in enumerate(recorders):
                    target_col = col_mapping[recorder]
                    if target_col not in df.columns:
                        continue

                    npy_path = os.path.join(audio_dir, recorder, npy_filename)
                    if not os.path.exists(npy_path):
                        continue

                    # Guard: clamp window range to actual audio duration
                    try:
                        audio_len_sec = int(np.load(npy_path, mmap_mode='r').shape[0] / 16000)
                    except Exception:
                        print(f"⚠️ [WARNING] Corrupted NPY, skipping: {npy_path}")
                        continue
                    effective_last = min(last_start_sec, audio_len_sec - window_size_sec)
                    if effective_last < 0:
                        continue

                    # Resolve optional embedding path
                    embed_path = None
                    if embedding_dir:
                        embed_path = _resolve_embedding_path(
                            embedding_dir, session, location, npy_stem, i + 1
                        )

                    # Convert to Python list of ints to avoid holding numpy/DataFrame references
                    labels = df[target_col].values.astype(int).tolist()

                    stride = test_stride_sec if session == test_session else train_stride_sec
                    if stride <= 0:
                        raise ValueError(f"Stride must be >= 1, got {stride}")

                    items = []
                    for sec in range(0, effective_last + 1, stride):
                        # Skip windows that overlap bus-obstructed frames
                        if valid_starts is not None and sec not in valid_starts:
                            bus_filtered_total += 1
                            continue
                        items.append({
                            'npy_path':       npy_path,
                            'start_sec':      sec,
                            'label':          labels[sec : sec + window_size_sec],
                            'session':        session,
                            'embedding_path': embed_path,  # None if no embedding available
                        })

                    if test_session == "all" or session == test_session:
                        test_list.extend(items)
                    else:
                        train_list.extend(items)
                            
    print(f"\n✅ Scan Complete! Maximized dataset prepared.")
    print(f"   -> Prepared {len(train_list):,} training windows (Stride: {train_stride_sec}s).")
    print(f"   -> Prepared {len(test_list):,} testing windows (Stride: {test_stride_sec}s, Hold-out: {test_session}).")
    if bus_filtered_total > 0:
        print(f"   -> Filtered {bus_filtered_total:,} windows containing bus-obstructed frames.")
    
    return train_list, test_list