import librosa.filters
import torch
import torch.nn as nn
import torchaudio


class SpecFlux(nn.Module):
    def __init__(self, sample_rate, device="cpu", n_fft=2048, win_length=1024, window='hann', center=False, pad_mode='reflect', eps=1e-10, lamb=0.1):
        super(SpecFlux, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = sample_rate // 100
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.eps = eps
        self.lamb = lamb
        self.filter_bank = torchaudio.transforms.MelScale(
            n_mels=82, sample_rate=sample_rate, n_stft=n_fft // 2 + 1, f_min=0.0, f_max=20000
        )
        self.window = torch.tensor(
            librosa.filters.get_window(window="hann", fftbins=False, Nx=win_length),
            requires_grad=False,
            dtype=torch.float32,
            device=device,
        )
        self.drum_mask = torch.ones(82, requires_grad=True, device=device)
        self.hihat_mask = torch.ones(82, requires_grad=True, device=device)
        self.snare_mask = torch.ones(82, requires_grad=True, device=device)

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
        spec = torch.log1p(spec * self.lamb)

        spec = self.filter_bank(spec)
        spec_diff = spec[..., 1:] - spec[..., :-1]
        spec_diff = torch.cat((torch.zeros_like(spec_diff[..., -1:]), spec_diff), dim=-1)

        drum_spec = torch.sum(spec_diff * self.drum_mask.unsqueeze(-1), dim=-2).unsqueeze(0)
        hihat_spec = torch.sum(spec_diff * self.hihat_mask.unsqueeze(-1), dim=-2).unsqueeze(0)
        snare_spec = torch.sum(spec_diff * self.snare_mask.unsqueeze(-1), dim=-2).unsqueeze(0)

        return torch.cat((drum_spec, snare_spec, hihat_spec), dim=0)


if __name__ == '__main__':
    spectral_flux = SpecFlux(sample_rate=48000, device="cpu")
