import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Vol


class Gain(torch.nn.Module):
    def __init__(self, min_gain: float = -20.0, max_gain: float = -1):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(self.min_gain, self.max_gain)
        audio = Vol(gain, gain_type="db")(audio)
        return audio


def load_audio(
    path: str | Path, sample_rate: int, normalize: bool
) -> torch.Tensor:
    audio, sr = torchaudio.load(path, normalize=True, backend="ffmpeg")
    audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(audio)
    audio = torch.mean(audio, dim=0, keepdim=False, dtype=torch.float32)
    if normalize:
        audio = audio / torch.max(torch.abs(audio))
    return audio

def get_indices(time_stamps: np.array, sample_rate: float, hop_size: int, fft_size: int) -> np.array:
    return (np.round((time_stamps * sample_rate) / hop_size + 0.5) - (np.ceil(fft_size / hop_size) - 1)).astype(int)