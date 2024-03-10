from copy import deepcopy

import librosa.filters
import torch
import torch.nn as nn
import torchaudio
import torch.nn.utils.parametrize as P


class ModelEmaV2(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class SpecFlux(nn.Module):
    def __init__(
            self,
            sample_rate,
            device="cpu",
            n_fft=2048,
            win_length=1024,
            window='hann',
            center=False,
            pad_mode='reflect',
            eps=1e-10,
            lamb=0.1,
            n_mels=82,
    ):
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
        self.n_mels = n_mels
        self.filter_bank = torchaudio.transforms.MelScale(
            n_mels=n_mels, sample_rate=sample_rate, n_stft=n_fft // 2 + 1, f_min=0.0, f_max=20000, norm=None, mel_scale="htk",
        )
        self.window = torch.tensor(
            librosa.filters.get_window(window="hann", fftbins=False, Nx=win_length),
            requires_grad=False,
            dtype=torch.float32,
            device=device,
        )
        self.activation = nn.ReLU()
        self.final_activation = nn.Softsign()
        self.feature_extractor = nn.Conv1d(in_channels=n_mels, out_channels=4, kernel_size=1, padding=0, bias=False)
        P.register_parametrization(self.feature_extractor, "weight", nn.LeakyReLU())
        self.drum_threshold = Threshold(mean_range=6, max_range=3, norm_range=8)
        self.snare_threshold = Threshold(mean_range=6, max_range=3, norm_range=8)
        self.hihat_threshold = Threshold(mean_range=6, max_range=3, norm_range=8)
        self.onset_threshold = Threshold(mean_range=6, max_range=3, norm_range=8)

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
        spec_diff = torch.clamp(spec_diff, min=0.0)
        spec_diff = torch.cat((torch.zeros_like(spec_diff[..., -1:]), spec_diff), dim=-1)

        # self.feature_extractor = nn.Conv1d(n_mels, 3, 1, padding=0, bias=False)
        features = self.feature_extractor(spec_diff)
        # features = self.activation(features)

        drum_spec = self.drum_threshold(features[:, 0, :].unsqueeze(-2))
        hihat_spec = self.hihat_threshold(features[:, 1, :].unsqueeze(-2))
        snare_spec = self.snare_threshold(features[:, 2, :].unsqueeze(-2))

        # spec_flux = torch.sum(spec_diff, dim=-2, keepdim=True)
        spec_flux = self.onset_threshold(features[:, 3, :].unsqueeze(-2))
        out = torch.cat((drum_spec, hihat_spec, snare_spec, spec_flux), dim=-2)
        # out = self.final_activation(out)
        return out


class Threshold(nn.Module):
    def __init__(self, channels=1, mean_range=1, max_range=1, norm_range=1, **kwargs):
        super(Threshold, self).__init__()
        self.threshold = nn.Conv1d(channels, channels, 1, padding=0, bias=True)
        P.register_parametrization(self.threshold, "weight", torch.nn.ReLU())
        P.register_parametrization(self.threshold, "bias", torch.nn.LeakyReLU())
        self.max = CausalMaxPool1d(kernel_size=max_range, **kwargs)
        self.mean = CausalAvgPool1d(kernel_size=mean_range, **kwargs)
        self.norm = CausalAvgPool1d(kernel_size=norm_range, **kwargs)
        self.Relu = nn.ReLU()

    def forward(self, x):
        mean = self.mean(x)
        maximums = self.max(x)
        norm = self.norm(x)
        threshold = self.threshold(norm) + mean
        mask = (x >= maximums) & (x >= threshold)
        x[~mask] = 0
        x[mask] = 1
        return x


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.causal_conv.padding[0] != 0:
            x = x[..., :-self.causal_conv.padding[0]]
        return x

class CausalMaxPool1d(nn.Module):
    def __init__(self, kernel_size, stride=1, dilation=1, **kwargs):
        super(CausalMaxPool1d, self).__init__()
        self.padding = ((kernel_size - 1) * dilation, 0)
        self.max_pool = nn.MaxPool1d(kernel_size, stride, padding=0, dilation=dilation, **kwargs)

    def forward(self, x):
        x = nn.functional.pad(x, self.padding)
        x = self.max_pool(x)
        return x

class CausalAvgPool1d(nn.Module):
    def __init__(self, kernel_size, stride=1, dilation=1, **kwargs):
        super(CausalAvgPool1d, self).__init__()
        self.padding = ((kernel_size - 1) * dilation, 0)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride, padding=0, **kwargs)

    def forward(self, x):
        x = nn.functional.pad(x, self.padding)
        x = self.avg_pool(x)
        return x


if __name__ == '__main__':
    spectral_flux = SpecFlux(sample_rate=48000, device="cpu")
