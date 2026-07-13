import torch
import torch.nn as nn
import torchaudio

class MelSpectrogramExtractor(nn.Module):
    """
    Converts 1D raw audio waveforms into 2D Log-Mel Spectrogram features.
    Configured precisely to match the VGGish feature extraction standards.
    """
    def __init__(self, sample_rate=16000, n_fft=512, win_length=512, 
                 hop_length=160, f_min=0.0, f_max=8000.0, n_mels=64):
        super(MelSpectrogramExtractor, self).__init__()
        
        # 1. Mel Spectrogram Transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0 # Power spectrogram
        )
        
        # 2. Convert Power to Log-Scale (Decibels)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Raw audio waveforms of shape (Batch, Num_Samples)
                              e.g., (Batch, 160000) for 10 seconds at 16kHz.
        Returns:
            torch.Tensor: Log-Mel features of shape (Batch, Seq_Len, n_mels)
        """
        # Output shape from mel_transform: (Batch, n_mels, Seq_Len)
        mel_spec = self.mel_transform(x)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        
        # Standardize features: mean = 0, std = 1
        mean = log_mel_spec.mean(dim=[1, 2], keepdim=True)
        std = log_mel_spec.std(dim=[1, 2], keepdim=True)
        norm_mel_spec = (log_mel_spec - mean) / (std + 1e-8)
        
        # Output shape: [Batch, 1, 64, Seq_Len]
        return norm_mel_spec.unsqueeze(1)

class Top1Pooler(nn.Module):
    """
    Extracts the Top-1 features across the temporal dimension.
    This effectively captures the most prominent acoustic event 
    (e.g., peak pedestrian noise) within a sliding window sequence.
    """
    def __init__(self):
        super(Top1Pooler, self).__init__()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence of shape (Batch, Seq_Len, Input_Dim)
            
        Returns:
            torch.Tensor: The Top-1 pooled features of shape (Batch, Input_Dim)
        """
        # Calculate the maximum value along the sequence length dimension (dim=1)
        pooled_features, _ = torch.max(x, dim=1)
        return pooled_features