"""extract_full_queries_va.py — Extract FULL 100-query embeddings for V.A.

Saves per-clip .pt files:
    {session}_{location}_{clip_id}_6m_ellipse_{rec_idx}_full_queries.pt
    → dict: {
        'queries': Tensor[T, 100, 256],    # all decoder queries per second
        'person_scores': Tensor[T, 100],   # person confidence per query
        'video_path': str,
        'feature_type': 'Full_100_Queries_256dim'
      }

Source: /media/chan/backup_SSD2/ASPED.a_6mVideo/{session}/{location}/Video/*.mp4
Output: /media/ykim/New Volume/ASPEDv1_VideoEmbeddings_full/
"""

import os
import re
import cv2
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_ROOT = "/media/chan/backup_SSD2/ASPED.a_6mVideo"
OUTPUT_DIR = "/media/ykim/New Volume/ASPEDv1_VideoEmbeddings_instance_full"

MODEL_NAME = "facebook/mask2former-swin-base-coco-instance"
PERSON_CLASS_ID = 0
BATCH_SIZE = 4        # lower than original to reduce memory pressure
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Match V.A filename pattern: 6m_ellipse_{rec}.mp4
VIDEO_PATTERN = re.compile(
    r'^6m_ellipse_(\d+)\.mp4$'
)


# ─────────────────────────────────────────────────────────────────────────────
# Extractor
# ─────────────────────────────────────────────────────────────────────────────

class FullQueryExtractor:
    """Extract all 100 query embeddings per frame from Mask2Former."""

    def __init__(self):
        print(f"[Extractor] Loading {MODEL_NAME} on {DEVICE}")
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.eval()
        self.hidden_dim = self.model.config.hidden_dim  # 256
        print(f"[Extractor] hidden_dim={self.hidden_dim}, ready.")

    @torch.no_grad()
    def extract_batch(self, bgr_frames: list):
        """Process BGR frames → queries [B, 100, 256], person_scores [B, 100]."""
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr_frames]
        inputs = self.processor(images=rgb_frames, return_tensors="pt").to(DEVICE)

        outputs = self.model(
            pixel_values=inputs.pixel_values,
            output_hidden_states=False,
        )
        # [B, 100, C] class logits, [B, 100, 256] hidden states
        class_logits = outputs.class_queries_logits
        hidden = outputs.transformer_decoder_last_hidden_state  # [B, 100, 256]

        probs = class_logits.softmax(dim=-1)          # [B, 100, C]
        person_scores = probs[:, :, PERSON_CLASS_ID]  # [B, 100]

        return hidden.cpu(), person_scores.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Frame reading (1 FPS from video)
# ─────────────────────────────────────────────────────────────────────────────

def read_frames_1fps(video_path: str) -> list:
    """Read video at 1 FPS. V.A videos are already at ~1 FPS or we subsample."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    if fps <= 1.5:
        # Already ~1 FPS
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    else:
        # Subsample to 1 FPS
        stride = max(1, round(fps))
        for i in range(0, total_frames, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

    cap.release()
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_va(extractor, sessions_filter=None):
    """Extract full query embeddings for all V.A clips."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sessions = sorted([s for s in os.listdir(VIDEO_ROOT)
                       if s.startswith("Session_")])
    if sessions_filter:
        sessions = [s for s in sessions if s in sessions_filter]

    print(f"Sessions to process: {sessions}")

    for session in sessions:
        sess_dir = os.path.join(VIDEO_ROOT, session)
        locations = sorted([d for d in os.listdir(sess_dir)
                            if os.path.isdir(os.path.join(sess_dir, d))])

        for location in locations:
            loc_dir = os.path.join(sess_dir, location)

            # V.A structure: Session/Location/ClipID/6m_ellipse_N.mp4
            clip_dirs = sorted([d for d in os.listdir(loc_dir)
                                if os.path.isdir(os.path.join(loc_dir, d))])

            for clip_id in clip_dirs:
                clip_dir = os.path.join(loc_dir, clip_id)
                mp4s = sorted([f for f in os.listdir(clip_dir) if f.endswith('.mp4')])

                for mp4_name in mp4s:
                    m = VIDEO_PATTERN.match(mp4_name)
                    if not m:
                        continue
                    rec_idx = m.group(1)

                    out_name = f"{session}_{location}_{clip_id}_6m_ellipse_{rec_idx}_full_queries.pt"
                    out_path = os.path.join(OUTPUT_DIR, out_name)

                    if os.path.exists(out_path):
                        print(f"  [SKIP] {out_name}")
                        continue

                    video_path = os.path.join(clip_dir, mp4_name)
                print(f"  {session}/{location}/{mp4_name} ...", end=" ", flush=True)

                try:
                    frames = read_frames_1fps(video_path)
                except Exception as e:
                    print(f"ERROR reading: {e}")
                    continue

                if not frames:
                    print("no frames")
                    continue

                # Extract in batches — save incrementally to avoid RAM blowup
                T = len(frames)
                # Pre-allocate on disk via memory-mapped tensor
                all_queries = torch.zeros(T, 100, 256)
                all_scores = torch.zeros(T, 100)
                idx = 0
                for i in range(0, T, BATCH_SIZE):
                    batch = frames[i:i + BATCH_SIZE]
                    queries, scores = extractor.extract_batch(batch)
                    bs = queries.shape[0]
                    all_queries[idx:idx+bs] = queries
                    all_scores[idx:idx+bs] = scores
                    idx += bs
                    # Free GPU memory
                    del queries, scores

                all_queries = all_queries[:idx]
                all_scores = all_scores[:idx]

                # Save
                torch.save({
                    'queries': all_queries,
                    'person_scores': all_scores,
                    'video_path': video_path,
                    'feature_type': 'Full_100_Queries_256dim',
                }, out_path)

                # Free RAM immediately
                del all_queries, all_scores

                size_mb = os.path.getsize(out_path) / 1e6
                print(f"{T}s, {size_mb:.0f}MB")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=str, nargs='*', default=None)
    args = parser.parse_args()

    extractor = FullQueryExtractor()
    extract_va(extractor, sessions_filter=args.session)

    print("\nDone!")
