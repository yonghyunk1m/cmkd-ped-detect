"""extract_embeddings.py — Extract Mask2Former 258-dim embeddings for V.B and V.C.

Produces per-clip .pt files with the same format as V.A ASPEDv1_VideoEmbeddings:
    {session}_{location}_{clip_id}_6m_ellipse_{rec_idx}_embed_258.pt
    → dict: {'embeddings': Tensor[T,258], 'video_path': str, 'feature_type': str}

258-dim layout: [256-dim Top-1 person query | max_confidence | gt_count_placeholder=0]
  - If no person detected → random no_person_embedding (same as V.A convention)
  - Temporal: frames averaged to 1 second

Datasets:
  V.B: Videos already at 1 FPS (pre-downsampled)
       /media/ykim/My Passport1/ASPEDvb/{session}/{location}/Video/{clip_id}.mp4
       Labels: /media/backup_SSD/ASPED_v2_npy/{session}/{location}/Labels/{clip_id}.csv
       col: recorder1_6m (single recorder)

  V.C: Raw GoPro videos at ~30 FPS, one per (session, location)
       Clip mapping: video_and_label_clip_metadata.csv → (filename, start_frame)
       Labels: /media/ykim/Linux/ASPED_v3_npy/{session}/{location}/Labels/{clip_id}.csv
       col: recorder1_6m, recorder2_6m, ... (multiple recorders)

Usage:
    python scripts/extract_embeddings.py --dataset vb
    python scripts/extract_embeddings.py --dataset vc
    python scripts/extract_embeddings.py --dataset vb --session Session_07262023
    python scripts/extract_embeddings.py --dataset vb --dry_run
"""

import os
import math
import argparse
import warnings
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

VB_VIDEO_ROOT  = "/media/ykim/My Passport1/ASPEDvb"
VB_LABEL_ROOT  = "/media/backup_SSD/ASPED_v2_npy"
VB_OUTPUT_DIR  = "/media/ykim/Linux/ASPEDv2_VideoEmbeddings"

VC_VIDEO_ROOT  = "/media/chan/backup_SSD2/ASPED.c"
VC_LABEL_ROOT  = "/media/ykim/Linux/ASPED_v3_npy"
VC_META_CSV    = "/media/chan/backup_SSD2/ASPED.c/video_and_label_clip_metadata.csv"
VC_OUTPUT_DIR  = "/media/ykim/Linux/ASPEDv3_VideoEmbeddings"

# Mask2Former
MODEL_NAME       = "facebook/mask2former-swin-base-coco-instance"
PERSON_CLASS_ID  = 0      # COCO-instance person label
CONF_THRESHOLD   = 0.5    # minimum confidence to count as person query
BATCH_SIZE       = 8      # frames per GPU batch (adjust to VRAM)
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Teacher model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class Mask2FormerExtractor:
    """Extract 258-dim per-frame embeddings using Mask2Former.

    For each frame:
      1. Run Mask2Former → class_queries_logits [100, C], hidden_state [100, 256]
      2. Find person queries above CONF_THRESHOLD
      3. Top-1 (highest person confidence) query → 256-dim embedding
      4. confidence score → scalar, gt_count → 0 (placeholder, filled from CSV at train time)
      5. No person → random no_person_embedding (sampled once at init, norm≈16)
    """

    def __init__(self):
        print(f"[Extractor] Loading {MODEL_NAME} on {DEVICE}")
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.eval()

        hidden_dim = self.model.config.hidden_dim  # 256
        self.hidden_dim = hidden_dim

        # no_person_embedding: fixed random vector (same convention as V.A)
        torch.manual_seed(42)
        self.no_person_emb = torch.randn(hidden_dim)
        self.no_person_emb = self.no_person_emb / self.no_person_emb.norm() * 16.0
        print(f"[Extractor] hidden_dim={hidden_dim}, no_person norm={self.no_person_emb.norm():.2f}")

    @torch.no_grad()
    def extract_batch(self, bgr_frames: list) -> torch.Tensor:
        """Process a list of BGR frames → [B, 258] tensor (CPU)."""
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr_frames]
        inputs = self.processor(images=rgb_frames, return_tensors="pt").to(DEVICE)

        outputs = self.model(
            pixel_values=inputs.pixel_values,
            output_hidden_states=False,
        )
        # [B, 100, C] class logits,  [B, 100, 256] hidden states
        class_logits = outputs.class_queries_logits          # [B, 100, C]
        hidden       = outputs.transformer_decoder_last_hidden_state  # [B, 100, 256]

        results = []
        for b in range(len(bgr_frames)):
            probs         = class_logits[b].softmax(dim=-1)   # [100, C]
            person_scores = probs[:, PERSON_CLASS_ID]          # [100]

            # Top-1 person query
            best_idx   = person_scores.argmax().item()
            best_score = person_scores[best_idx].item()

            if best_score >= CONF_THRESHOLD:
                feat = hidden[b, best_idx]                     # [256]
            else:
                feat = self.no_person_emb.to(DEVICE)

            # 258-dim: [256 feat | confidence | 0 placeholder]
            vec = torch.cat([feat.cpu(),
                             torch.tensor([best_score, 0.0])], dim=0)  # [258]
            results.append(vec)

        return torch.stack(results, dim=0)  # [B, 258]


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_frames_1fps(video_path: str) -> list:
    """Read a 1-FPS video and return all frames (already 1 frame/sec)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def read_frames_at_1fps(video_path: str, start_frame: int, n_seconds: int) -> list:
    """From a ~30fps video, read 1 frame per second starting at start_frame.

    Returns list of n_seconds BGR frames (may be shorter if video ends).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride       = max(1, round(fps))  # frames per second

    frames = []
    for s in range(n_seconds):
        target = start_frame + s * stride
        if target >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def embed_frames(extractor: Mask2FormerExtractor, frames: list) -> torch.Tensor:
    """Run extractor over frames in batches → [T, 258]."""
    if not frames:
        return torch.zeros(0, 258)
    all_vecs = []
    for i in range(0, len(frames), BATCH_SIZE):
        batch = frames[i : i + BATCH_SIZE]
        vecs  = extractor.extract_batch(batch)   # [B, 258]
        all_vecs.append(vecs)
    return torch.cat(all_vecs, dim=0)            # [T, 258]


# ─────────────────────────────────────────────────────────────────────────────
# V.B extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_vb(extractor: Mask2FormerExtractor, sessions_filter=None, dry_run=False):
    """Extract 258-dim embeddings for all V.B clips."""
    os.makedirs(VB_OUTPUT_DIR, exist_ok=True)

    sessions = sorted([s for s in os.listdir(VB_VIDEO_ROOT)
                       if s.startswith("Session_") and not s.endswith(".md")])
    if sessions_filter:
        sessions = [s for s in sessions if s in sessions_filter]

    for session in sessions:
        sess_video = os.path.join(VB_VIDEO_ROOT, session)
        sess_label = os.path.join(VB_LABEL_ROOT, session)

        locations = sorted([d for d in os.listdir(sess_video)
                            if os.path.isdir(os.path.join(sess_video, d))])

        for location in locations:
            video_dir = os.path.join(sess_video, location, "Video")
            label_dir = os.path.join(sess_label, location, "Labels")
            if not os.path.exists(video_dir) or not os.path.exists(label_dir):
                print(f"  [skip] {session}/{location}: missing video or label dir")
                continue

            clip_ids = sorted([f.replace(".csv", "")
                               for f in os.listdir(label_dir) if f.endswith(".csv")])

            for clip_id in tqdm(clip_ids, desc=f"{session}/{location}", leave=False):
                video_path = os.path.join(video_dir, f"{clip_id}.mp4")
                if not os.path.exists(video_path):
                    # try uppercase
                    video_path = os.path.join(video_dir, f"{clip_id}.MP4")
                if not os.path.exists(video_path):
                    print(f"  [skip] no video: {video_path}")
                    continue

                # V.B has only recorder1
                out_fname = f"{session}_{location}_{clip_id}_6m_ellipse_1_embed_258.pt"
                out_path  = os.path.join(VB_OUTPUT_DIR, out_fname)
                if os.path.exists(out_path):
                    continue

                if dry_run:
                    print(f"  [dry] would embed: {out_fname}")
                    continue

                frames = read_frames_1fps(video_path)
                embs   = embed_frames(extractor, frames)  # [T, 258]

                torch.save({
                    "embeddings":   embs,
                    "video_path":   video_path,
                    "feature_type": "Top1_Person_with_Metadata_258dim",
                }, out_path)

    print(f"\n[V.B] Done. Output: {VB_OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# V.C extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_vc(extractor: Mask2FormerExtractor, sessions_filter=None, dry_run=False):
    """Extract 258-dim embeddings for all V.C clips."""
    os.makedirs(VC_OUTPUT_DIR, exist_ok=True)

    meta = pd.read_csv(VC_META_CSV)
    # meta columns: session, location, camera, filename, start_frame, label_file

    sessions = sorted(meta["session"].unique())
    if sessions_filter:
        sessions = [s for s in sessions if s in sessions_filter]

    for session in sessions:
        sess_label = os.path.join(VC_LABEL_ROOT, session)
        sess_meta  = meta[meta["session"] == session]

        for _, row in sess_meta.iterrows():
            location    = row["location"]
            camera      = row["camera"]
            filename    = row["filename"]
            start_frame = int(row["start_frame"])

            # Find the gopro video file
            gopro_dir  = os.path.join(VC_VIDEO_ROOT, session, location, "Video", camera)
            video_path = os.path.join(gopro_dir, filename + ".MP4")
            if not os.path.exists(video_path):
                video_path = os.path.join(gopro_dir, filename + ".mp4")
            if not os.path.exists(video_path):
                print(f"  [skip] no video: {gopro_dir}/{filename}.MP4")
                continue

            label_dir = os.path.join(sess_label, location, "Labels")
            if not os.path.exists(label_dir):
                print(f"  [skip] no labels: {label_dir}")
                continue

            clip_ids = sorted([f.replace(".csv", "")
                               for f in os.listdir(label_dir) if f.endswith(".csv")])

            # Count total seconds across all clips (to know how many frames to extract)
            total_seconds = 0
            for cid in clip_ids:
                csv_path = os.path.join(label_dir, f"{cid}.csv")
                try:
                    df = pd.read_csv(csv_path)
                    total_seconds += len(df)
                except Exception:
                    pass

            if total_seconds == 0:
                continue

            # Check if all output files already exist
            all_done = True
            for cid in clip_ids:
                csv_path = os.path.join(label_dir, f"{cid}.csv")
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    continue
                rec_cols = [c for c in df.columns if c.endswith("_6m") and not c.startswith("view_")]
                for col in rec_cols:
                    rec_idx = col.replace("recorder", "").replace("_6m", "")
                    out_fname = f"{session}_{location}_{cid}_6m_ellipse_{rec_idx}_embed_258.pt"
                    if not os.path.exists(os.path.join(VC_OUTPUT_DIR, out_fname)):
                        all_done = False
                        break
                if not all_done:
                    break

            if all_done:
                continue

            if dry_run:
                print(f"  [dry] would embed {session}/{location} ({total_seconds}s from {filename}+{start_frame})")
                continue

            # Extract all frames at once from the single gopro video
            print(f"  Extracting {total_seconds}s from {video_path} (start_frame={start_frame})")
            frames = read_frames_at_1fps(video_path, start_frame, total_seconds)

            if not frames:
                print(f"  [skip] no frames extracted")
                continue

            # Embed all frames: [total_T, 258]
            all_embs = embed_frames(extractor, frames)

            # Slice by clip and save per (clip_id, recorder)
            cursor = 0
            for cid in clip_ids:
                csv_path = os.path.join(label_dir, f"{cid}.csv")
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    continue
                clip_len = len(df)
                clip_embs = all_embs[cursor : cursor + clip_len]  # [clip_len, 258]
                cursor += clip_len

                rec_cols = [c for c in df.columns if c.endswith("_6m") and not c.startswith("view_")]
                for col in rec_cols:
                    rec_idx = col.replace("recorder", "").replace("_6m", "")
                    # V.C uses the same video for all recorders (video is scene-level, not per-recorder)
                    # Each recorder sees the same video embedding but has different audio GT
                    out_fname = f"{session}_{location}_{cid}_6m_ellipse_{rec_idx}_embed_258.pt"
                    out_path  = os.path.join(VC_OUTPUT_DIR, out_fname)
                    if os.path.exists(out_path):
                        continue
                    torch.save({
                        "embeddings":   clip_embs,
                        "video_path":   video_path,
                        "feature_type": "Top1_Person_with_Metadata_258dim",
                    }, out_path)

    print(f"\n[V.C] Done. Output: {VC_OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  choices=["vb", "vc", "both"], default="both")
    parser.add_argument("--session",  nargs="+", default=None,
                        help="Filter to specific sessions (e.g. Session_07262023)")
    parser.add_argument("--dry_run",  action="store_true",
                        help="List what would be done without running Mask2Former")
    args = parser.parse_args()

    sessions_filter = set(args.session) if args.session else None

    extractor = None if args.dry_run else Mask2FormerExtractor()

    if args.dataset in ("vb", "both"):
        print("\n" + "="*60)
        print("V.B Embedding Extraction")
        print("="*60)
        extract_vb(extractor, sessions_filter=sessions_filter, dry_run=args.dry_run)

    if args.dataset in ("vc", "both"):
        print("\n" + "="*60)
        print("V.C Embedding Extraction")
        print("="*60)
        extract_vc(extractor, sessions_filter=sessions_filter, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
