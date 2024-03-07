import librosa.filters
import torch
import torch.nn as nn
import torchaudio


class SpecFlux(nn.Module):
    def __init__(self, sample_rate, n_fft=2048, win_length=1024, window='hann', center=False, pad_mode='reflect', eps=1e-10):
        super(SpecFlux, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = sample_rate // 100
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.eps = eps
        self.filter_bank = torchaudio.transforms.MelScale(
            n_mels=82, sample_rate=sample_rate, n_stft=n_fft // 2 + 1, f_min=0.0, f_max=20000
        )
        self.window = torch.tensor(
            librosa.filters.get_window(window="hann", fftbins=False, Nx=win_length),
            requires_grad=False,
            dtype=torch.float32,
        )
        self.drum_mask = torch.ones(82, requires_grad=True)
        self.hihat_mask = torch.ones(82, requires_grad=True)
        self.snare_mask = torch.ones(82, requires_grad=True)

    def forward(self, x):

        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            center=self.center,
            pad_mode=self.pad_mode,
            return_complex=True,
        )
        spec = abs(spec)

        spec = self.filter_bank(spec)
        spec_diff = spec[..., 1:] - spec[..., :-1]
        spec_diff = torch.clamp(spec_diff, min=self.eps)
        spec_diff = torch.log(spec_diff)
        drum_spec = spec_diff * self.drum_mask.unsqueeze(-1)
        hihat_spec = spec_diff * self.hihat_mask.unsqueeze(-1)
        snare_spec = spec_diff * self.snare_mask.unsqueeze(-1)

        return drum_spec, hihat_spec, snare_spec


if __name__ == '__main__':
    spectral_flux = SpecFlux(sample_rate=48000)
