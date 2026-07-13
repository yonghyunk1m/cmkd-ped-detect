import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import math

# (comment removed)
SOURCE_ROOT = '/media/chan/backup_SSD2/ASPED.c'
DEST_ROOT = '/media/backup_SSD/ASPED_v3_npy'

TARGET_SR = 16000
CHUNK_HOURS = 4
CHUNK_SECONDS = CHUNK_HOURS * 3600
CHUNK_SAMPLES = TARGET_SR * CHUNK_SECONDS
AUDIO_EXT = '.wav'
# ========================================

def check_npy_validity(npy_path, expected_samples):
    """Utility function."""
    if not os.path.exists(npy_path):
        return False
    try:
        data = np.load(npy_path, mmap_mode='r')
        if data.shape[0] >= expected_samples * 0.995:
            return True
        return False
    except:
        return False

def get_valid_wav_files(rec_source_path):
    """Utility function."""
    all_files = sorted([f for f in os.listdir(rec_source_path) 
                        if f.endswith(AUDIO_EXT) and not f.startswith('.')])
    if not all_files: return []

    trimmed_indices = [i for i, f in enumerate(all_files) if '_trimmed' in f]
    
    # (comment removed)
    if not trimmed_indices: return all_files

    start_idx = trimmed_indices[0]
    
    # (comment removed)
    if len(trimmed_indices) == 1:
        # (comment removed)
        end_idx = len(all_files) - 1
    else:
        # (comment removed)
        end_idx = trimmed_indices[-1]

    in_range_files = all_files[start_idx : end_idx + 1]
    
    final_files = []
    for f in in_range_files:
        if '_trimmed' not in f:
            if f.replace('.wav', '_trimmed.wav') in in_range_files:
                continue
        final_files.append(f)
        
    return final_files

def process_session_folder(session_path, dest_session_path):
    audio_root = os.path.join(session_path, 'Audio')
    label_root = os.path.join(session_path, 'Labels')

    if not os.path.exists(audio_root) or not os.path.exists(label_root): 
        return

    label_files = [f for f in os.listdir(label_root) if f.endswith('_clipped.csv')]
    if not label_files: 
        return
    
    df_labels = pd.read_csv(os.path.join(label_root, label_files[0]))
    total_label_seconds = len(df_labels)
    recorder_dirs = sorted([d for d in os.listdir(audio_root) if os.path.isdir(os.path.join(audio_root, d))])
    
    for idx, recorder_folder in enumerate(recorder_dirs):
        mapped_id = idx + 1
        # (comment removed)
        if not any(col.startswith(f"recorder{mapped_id}_") for col in df_labels.columns):
            continue

        rec_source_path = os.path.join(audio_root, recorder_folder)
        rec_dest_path = os.path.join(dest_session_path, 'Audio', f"recorder_{recorder_folder}")
        os.makedirs(rec_dest_path, exist_ok=True)

        total_chunks = math.ceil(total_label_seconds / CHUNK_SECONDS)
        wav_files = get_valid_wav_files(rec_source_path)
        
        for i in range(total_chunks):
            npy_name = f"{i+1:04d}.npy"
            npy_path = os.path.join(rec_dest_path, npy_name)
            csv_name = f"{i+1:04d}.csv"
            
            chunk_duration = min(CHUNK_SECONDS, total_label_seconds - (i * CHUNK_SECONDS))
            expected_samples = chunk_duration * TARGET_SR
            
            # (comment removed)
            if check_npy_validity(npy_path, expected_samples):
                continue
            
            print(f"  >> Extracting: {recorder_folder} -> {npy_name} ({chunk_duration}s)")
            
            chunk_audio_parts = []
            current_pos = 0 
            target_start = i * CHUNK_SAMPLES
            target_end = target_start + expected_samples 

            for wav in tqdm(wav_files, leave=False, desc=f"Chunk {i+1}"):
                wav_p = os.path.join(rec_source_path, wav)
                info = torchaudio.info(wav_p)
                wav_len_samples = int(info.num_frames * TARGET_SR / info.sample_rate)
                
                wav_start = current_pos
                wav_end = current_pos + wav_len_samples
                
                # (comment removed)
                if wav_end > target_start and wav_start < target_end:
                    waveform, sr = torchaudio.load(wav_p)
                    if sr != TARGET_SR:
                        waveform = T.Resample(sr, TARGET_SR)(waveform)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    # (comment removed)
                    crop_start = max(0, target_start - wav_start)
                    crop_end = min(wav_len_samples, target_end - wav_start)
                    
                    chunk_audio_parts.append(waveform[:, crop_start:crop_end])
                
                current_pos += wav_len_samples
                if current_pos >= target_end: 
                    break

            # (comment removed)
            if chunk_audio_parts:
                full_chunk = torch.cat(chunk_audio_parts, dim=1).squeeze().numpy()
                np.save(npy_path, full_chunk.astype(np.float32))

            # (comment removed)
            dest_label_path = os.path.join(dest_session_path, 'Labels')
            os.makedirs(dest_label_path, exist_ok=True)
            save_csv_path = os.path.join(dest_label_path, csv_name)
            
            chunk_df = df_labels.iloc[i*CHUNK_SECONDS : (i+1)*CHUNK_SECONDS]
            chunk_df.to_csv(save_csv_path, index=False)

            # (comment removed)
            if os.path.exists(npy_path) and os.path.exists(save_csv_path):
                verify_data = np.load(npy_path, mmap_mode='r')
                audio_len_sec = verify_data.shape[0] / TARGET_SR
                csv_len_sec = len(chunk_df)
                
                if abs(audio_len_sec - csv_len_sec) > 1.0:
                    print(f"\n    🚨 [Verification Failed] Mismatch in {npy_name}!")
                    print(f"       - Audio: {audio_len_sec:.1f}s | Label: {csv_len_sec}s")
                    print(f"       - Corrupted data detected. Deleting files...")
                    os.remove(npy_path)
                    os.remove(save_csv_path)

def main():
    if not os.path.exists(DEST_ROOT): 
        os.makedirs(DEST_ROOT)
    for root, dirs, files in os.walk(SOURCE_ROOT):
        if 'Audio' in dirs and 'Labels' in dirs:
            process_session_folder(root, os.path.join(DEST_ROOT, os.path.relpath(root, SOURCE_ROOT)))

if __name__ == '__main__':
    main()