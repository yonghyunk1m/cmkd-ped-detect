import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random

# Module-level lazy cache: embedding_path → full tensor.
# Pre-populate in the main process via preload_all_embeddings() before spawning
# DataLoader workers — Linux fork() workers inherit the populated cache via CoW,
# eliminating all embedding disk I/O after the initial load.
_EMBED_CACHE: dict = {}


def preload_all_embeddings(embed_dir: str) -> None:
    """Load every *_embed_258.pt file in embed_dir into _EMBED_CACHE.

    Call this once in the main process before creating any DataLoader.
    Workers forked afterwards inherit the cache via copy-on-write, so no
    further disk reads are needed for embeddings during training.
    """
    import glob
    from tqdm import tqdm

    files = sorted(glob.glob(os.path.join(embed_dir, '*_embed_258.pt')))
    if not files:
        print(f"[preload] No *_embed_258.pt files found in {embed_dir}")
        return

    already = len(_EMBED_CACHE)
    to_load = [f for f in files if f not in _EMBED_CACHE]
    if not to_load:
        print(f"[preload] All {len(files)} files already in cache.")
        return

    print(f"[preload] Loading {len(to_load)} embedding files into RAM "
          f"({len(already) if already else 0} already cached)…")
    for path in tqdm(to_load, desc="Pre-loading embeddings", unit="file"):
        _load_embedding_tensor(path)

    total_gb = sum(v.element_size() * v.nelement() for v in _EMBED_CACHE.values()) / 1e9
    print(f"[preload] Done. {len(_EMBED_CACHE)} files cached, {total_gb:.2f} GB in RAM.")


def _load_embedding_tensor(embedding_path: str) -> torch.Tensor:
    """Load and cache a Mask2Former embedding file.

    Expected .pt content:
      dict with key 'embeddings': Tensor[total_seconds * N_queries, D]
      OR a bare Tensor of the same shape.
    """
    if embedding_path not in _EMBED_CACHE:
        raw = torch.load(embedding_path, map_location='cpu', weights_only=False)
        if isinstance(raw, dict) and 'embeddings' in raw:
            raw = raw['embeddings']
        _EMBED_CACHE[embedding_path] = raw
    return _EMBED_CACHE[embedding_path]


class ASPEDDataset(Dataset):
    def __init__(self, data_list, task_type='classification', max_class=3):
        self.data_list = data_list
        self.task_type = task_type
        self.max_class = max_class
        self.SR = 16000 

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if not self.data_list:
            raise IndexError("ASPEDDataset is empty.")

        max_attempts = min(20, len(self.data_list))
        last_error = None
        fallback_item = self.data_list[idx]

        for attempt in range(max_attempts):
            current_idx = idx if attempt == 0 else random.randint(0, len(self.data_list) - 1)
            item = self.data_list[current_idx]
            try:
                # 1. mmap disk mapping
                features_mmap = np.load(item['npy_path'], mmap_mode='r')
                total_len = features_mmap.shape[0]

                start_sec = item['start_sec']
                start_idx = int(start_sec * self.SR)
                end_idx = start_idx + int(10 * self.SR)

                valid_length = total_len - start_idx
                if valid_length < self.SR:
                    print(f"\n🚨 [Short Audio Skipped] File: {item['npy_path']}")
                    print(f"   - Start: {start_sec}s, Valid Length: {max(0, valid_length)/self.SR:.2f}s")
                    continue

                window_audio = np.zeros((int(10 * self.SR),), dtype=np.float32)
                actual_end = min(end_idx, total_len)
                slice_data = features_mmap[start_idx:actual_end]
                window_audio[:slice_data.shape[0]] = slice_data

                l_orig = [min(int(l), self.max_class) for l in item['label']]
                if len(l_orig) < 10:
                    l_orig = l_orig + [0] * (10 - len(l_orig))
                label = torch.tensor(l_orig[:10], dtype=torch.long)
                return torch.from_numpy(window_audio), label

            except Exception as e:
                last_error = e
                print(f"\n❌ [File Error] Failed to load {item['npy_path']}: {e}")
                continue

        # Avoid infinite retry loops: return zero-audio fallback after bounded attempts.
        print(f"\n⚠️ [Data Fallback] Using zero audio after {max_attempts} failed attempts. Last error: {last_error}")
        fallback_audio = np.zeros((int(10 * self.SR),), dtype=np.float32)
        l_orig = [min(int(l), self.max_class) for l in fallback_item['label']]
        if len(l_orig) < 10:
            l_orig = l_orig + [0] * (10 - len(l_orig))
        fallback_label = torch.tensor(l_orig[:10], dtype=torch.long)
        return torch.from_numpy(fallback_audio), fallback_label


class ASPEDDatasetKD(ASPEDDataset):
    """ASPEDDataset extended with Mask2Former teacher embeddings.

    Each data_list item must contain an 'embedding_path' key pointing to a
    .pt file with pre-extracted Mask2Former query embeddings:
        Tensor[total_seconds * N_queries, D_teacher]

    __getitem__ returns:
        audio   : Tensor[160_000]          float32
        emb     : Tensor[10, N_q, D]       float32   (10 seconds × N_q queries × D dim)
        label   : Tensor[10]               long

    Items without a valid 'embedding_path' are skipped (treated like corrupted audio).
    """

    N_QUERIES = 100  # Mask2Former outputs 100 queries per frame

    def __init__(self, data_list, task_type='classification', max_class=3,
                 strip_metadata=False):
        # Filter out items that lack a usable embedding file upfront so that
        # __len__ and WeightedRandomSampler see a consistent size.
        valid = [
            item for item in data_list
            if item.get('embedding_path') and os.path.exists(item['embedding_path'])
        ]
        if len(valid) < len(data_list):
            print(f"[ASPEDDatasetKD] Filtered {len(data_list) - len(valid)} items "
                  f"without embeddings. Remaining: {len(valid)}")
        super().__init__(valid, task_type=task_type, max_class=max_class)
        self.strip_metadata = strip_metadata

    def _load_embedding_window(self, embedding_path: str, start_sec: int) -> torch.Tensor:
        """Extract the 10-second embedding window.

        Handles two file formats automatically:
          Pre-aggregated  [T, D]       e.g. _embed_258.pt  → returns [10, D]
          Raw queries     [T*N_q, D]   e.g. local 256-dim  → returns [10, N_q, D]

        Detection: if total rows are a multiple of N_QUERIES AND much larger than
        expected seconds, treat as raw-query format.
        """
        emb = _load_embedding_tensor(embedding_path)
        D   = emb.shape[-1]
        T   = emb.shape[0]

        # Pre-aggregated files: T ≤ ~14400 (4-hour recording in seconds)
        # Raw-query files:      T = seconds * 100 ≥ 100_000
        is_raw_queries = T > 100_000

        if is_raw_queries:
            # [T*N_q, D] → [10, N_q, D]
            s = start_sec * self.N_QUERIES
            e = (start_sec + 10) * self.N_QUERIES
            available = T - s
            if available <= 0:
                return torch.zeros(10, self.N_QUERIES, D)
            window = emb[s:e]
            if window.shape[0] < 10 * self.N_QUERIES:
                pad = torch.zeros(10 * self.N_QUERIES - window.shape[0], D)
                window = torch.cat([window, pad], dim=0)
            return window.view(10, self.N_QUERIES, D)
        else:
            # [T, D] pre-aggregated → [10, D]
            s = start_sec
            e = start_sec + 10
            available = T - s
            if available <= 0:
                return torch.zeros(10, D)
            window = emb[s:e]
            if window.shape[0] < 10:
                pad = torch.zeros(10 - window.shape[0], D)
                window = torch.cat([window, pad], dim=0)
            return window  # [10, D]

    def __getitem__(self, idx):
        if not self.data_list:
            raise IndexError("ASPEDDatasetKD is empty.")

        max_attempts = min(20, len(self.data_list))
        fallback_item = self.data_list[idx]

        for attempt in range(max_attempts):
            current_idx = idx if attempt == 0 else random.randint(0, len(self.data_list) - 1)
            item = self.data_list[current_idx]
            embedding_path = item.get('embedding_path', '')

            try:
                # 1. Load audio (same logic as parent)
                features_mmap = np.load(item['npy_path'], mmap_mode='r')
                total_len = features_mmap.shape[0]
                start_sec = item['start_sec']
                start_idx = int(start_sec * self.SR)
                end_idx   = start_idx + int(10 * self.SR)

                if (total_len - start_idx) < self.SR:
                    continue

                window_audio = np.zeros((int(10 * self.SR),), dtype=np.float32)
                actual_end   = min(end_idx, total_len)
                window_audio[:actual_end - start_idx] = features_mmap[start_idx:actual_end]

                # 2. Load label
                l_orig = [min(int(l), self.max_class) for l in item['label']]
                if len(l_orig) < 10:
                    l_orig = l_orig + [0] * (10 - len(l_orig))
                label = torch.tensor(l_orig[:10], dtype=torch.long)

                # 3. Load embedding window
                emb = self._load_embedding_window(embedding_path, start_sec)
                if self.strip_metadata:
                    emb = emb[..., :256]

                return torch.from_numpy(window_audio), emb, label

            except Exception as e:
                print(f"\n[ASPEDDatasetKD] Failed idx={current_idx}: {e}")
                continue

        # Fallback: return zero tensors
        print(f"\n[ASPEDDatasetKD] Fallback after {max_attempts} attempts (idx={idx})")
        fallback_audio = torch.zeros(int(10 * self.SR))
        fallback_emb   = torch.zeros(10, self.N_QUERIES, 256)
        l_orig = [min(int(l), self.max_class) for l in fallback_item['label']]
        if len(l_orig) < 10:
            l_orig = l_orig + [0] * (10 - len(l_orig))
        fallback_label = torch.tensor(l_orig[:10], dtype=torch.long)
        return fallback_audio, fallback_emb, fallback_label