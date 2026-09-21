"""audio_encoders.py — Unified audio encoder wrappers for VGGish, CED, BEATs, PaSST.

All encoders: input [B, 160000] (10s @ 16kHz) → output [B, 10, D]
              (10 per-second embeddings, D varies by encoder)

Usage:
    encoder = build_encoder('ced')   # or 'vggish', 'beats', 'passt'
    out = encoder(audio)             # [B, 10, D]
    print(encoder.output_dim)        # 768 for CED/BEATs/PaSST, 512 for VGGish
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGishEncoder(nn.Module):
    """Wrapper around existing VGGish for unified interface."""

    def __init__(self):
        super().__init__()
        from .backbone import MelSpectrogramExtractor
        from .vggish import VGGish
        self.mel_transform = MelSpectrogramExtractor(n_mels=64)
        self.model = VGGish(pretrained=True, freeze=True, per_second=True)
        self.output_dim = 512

    def forward(self, audio):
        """audio: [B, 160000] → [B, 10, 512]"""
        mel = self.mel_transform(audio)          # [B, 1, 64, ~1000]
        feat = self.model(mel)                    # [B, 512, 10]
        return feat.permute(0, 2, 1)              # [B, 10, 512]

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self


class CEDEncoder(nn.Module):
    """CED (Consistent Ensemble Distillation) from HuggingFace."""

    MODEL_NAME = 'mispeech/ced-base'

    def __init__(self, finetune=False):
        super().__init__()
        from transformers import AutoModel, AutoFeatureExtractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True)
        self.finetune = finetune
        if not finetune:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        self.output_dim = 768

    def _audio_to_mel(self, audio):
        """Convert raw audio tensor to mel input expected by CED.
        audio: [B, 160000] → [B, 64, 1001]
        Uses same torchaudio transforms as CED's feature_extractor internally,
        but computed on GPU in batch for speed.
        """
        device = audio.device
        if not hasattr(self, '_mel_transform'):
            import torchaudio.transforms as T
            self._mel_transform = T.MelSpectrogram(
                f_min=0, sample_rate=16000, win_length=512, center=True,
                n_fft=512, f_max=None, hop_length=160, n_mels=64,
            ).to(device)
            self._amp_to_db = T.AmplitudeToDB(top_db=120).to(device)
        mel = self._mel_transform(audio.float())  # [B, 64, T]
        mel = self._amp_to_db(mel)                 # [B, 64, T] raw dB, no normalization
        # Pad/trim to 1001 (CED expected length for 10s@16kHz)
        target = 1001
        if mel.shape[-1] > target:
            mel = mel[..., :target]
        elif mel.shape[-1] < target:
            mel = torch.nn.functional.pad(mel, (0, target - mel.shape[-1]))
        return mel

    def forward(self, audio):
        """audio: [B, 160000] → [B, 10, 768]"""
        mel = self._audio_to_mel(audio)             # [B, 64, 1001]
        with torch.no_grad():
            out = self.model(mel)                    # logits: [B, T, 768]
        hidden = out.logits                          # [B, ~248, 768]
        # Pool to 10 timesteps (1 per second)
        # Use max pool instead of avg pool to preserve salient features
        # (avg pool over 24 frames destroys short events like footsteps)
        B, T, D = hidden.shape
        pooled = F.adaptive_max_pool1d(
            hidden.permute(0, 2, 1), 10).permute(0, 2, 1)  # [B, 10, 768]
        return pooled

    def train(self, mode=True):
        super().train(mode)
        if not self.finetune:
            self.model.eval()
        return self


class BEATsEncoder(nn.Module):
    """BEATs audio encoder from HuggingFace/SpeechBrain."""

    def __init__(self):
        super().__init__()
        try:
            from transformers import AutoModel, AutoFeatureExtractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                'microsoft/BEATs-base', trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                'microsoft/BEATs-base', trust_remote_code=True)
        except Exception:
            # Fallback: try BEATs from speechbrain or direct download
            raise RuntimeError(
                "BEATs not available via transformers. "
                "Install from: https://github.com/microsoft/unilm/tree/master/beats"
            )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.output_dim = 768

    def forward(self, audio):
        """audio: [B, 160000] → [B, 10, 768]"""
        device = audio.device
        batch_inputs = []
        for i in range(audio.shape[0]):
            wav = audio[i].cpu().numpy()
            inputs = self.feature_extractor(
                wav, sampling_rate=16000, return_tensors='pt')
            batch_inputs.append(inputs['input_values'].squeeze(0))
        input_values = torch.stack(batch_inputs).to(device)

        with torch.no_grad():
            out = self.model(input_values)
        hidden = out.last_hidden_state  # [B, T, 768]
        B, T, D = hidden.shape
        pooled = F.adaptive_avg_pool1d(
            hidden.permute(0, 2, 1), 10).permute(0, 2, 1)
        return pooled

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self


class PaSSTEncoder(nn.Module):
    """PaSST (Patchout fASt Spectrogram Transformer)."""

    def __init__(self):
        super().__init__()
        try:
            from transformers import AutoModel, AutoFeatureExtractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                'fschmid56/passt-base', trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                'fschmid56/passt-base', trust_remote_code=True)
        except Exception:
            raise RuntimeError(
                "PaSST not available via transformers. "
                "Install from: https://github.com/kkoutini/PaSST"
            )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.output_dim = 768

    def forward(self, audio):
        """audio: [B, 160000] → [B, 10, 768]"""
        device = audio.device
        batch_inputs = []
        for i in range(audio.shape[0]):
            wav = audio[i].cpu().numpy()
            inputs = self.feature_extractor(
                wav, sampling_rate=16000, return_tensors='pt')
            batch_inputs.append(inputs['input_values'].squeeze(0))
        input_values = torch.stack(batch_inputs).to(device)

        with torch.no_grad():
            out = self.model(input_values)
        hidden = out.last_hidden_state  # [B, T, 768]
        B, T, D = hidden.shape
        pooled = F.adaptive_avg_pool1d(
            hidden.permute(0, 2, 1), 10).permute(0, 2, 1)
        return pooled

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

ENCODERS = {
    'vggish': VGGishEncoder,
    'ced': CEDEncoder,
    'beats': BEATsEncoder,
    'passt': PaSSTEncoder,
}


def build_encoder(name: str, finetune: bool = False) -> nn.Module:
    """Build audio encoder by name. Returns module with .output_dim attribute."""
    name = name.lower()
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder '{name}'. Choose from: {list(ENCODERS.keys())}")
    print(f"[Encoder] Building '{name}' audio encoder (finetune={finetune})...")
    if name == 'ced':
        encoder = ENCODERS[name](finetune=finetune)
    else:
        encoder = ENCODERS[name]()
    print(f"[Encoder] {name}: output_dim={encoder.output_dim}")
    return encoder
