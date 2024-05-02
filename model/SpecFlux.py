import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

from model import CausalMaxPool1d, CausalAvgPool1d


class SpecFlux(nn.Module):
    def __init__(
        self,
        eps=1e-10,
        lamb=0.1,
        n_mels=82,
        threshold=True,
    ):
        super(SpecFlux, self).__init__()
        self.eps = eps
        self.lamb = torch.tensor(lamb, requires_grad=True)
        self.n_mels = n_mels
        self.activation = nn.ELU()
        self.final_activation = nn.Tanh()
        self.feature_extractor = nn.Conv1d(
            in_channels=n_mels, out_channels=4, kernel_size=1, padding=0, bias=False
        )
        P.register_parametrization(self.feature_extractor, "weight", nn.SELU())
        self.drum_threshold = Threshold(mean_range=6, max_range=3, norm_range=8)
        self.snare_threshold = Threshold(mean_range=6, max_range=3, norm_range=8)
        self.hihat_threshold = Threshold(mean_range=6, max_range=3, norm_range=8)
        self.onset_threshold = Threshold(mean_range=6, max_range=3, norm_range=8)
        self.threshold = threshold

    def forward(self, x):
        mel = torch.log1p(x * self.lamb)
        spec_diff = mel[..., 1:] - mel[..., :-1]
        spec_diff = torch.clamp(spec_diff, min=0.0)
        spec_diff = torch.cat(
            (torch.zeros_like(spec_diff[..., -1:]), spec_diff), dim=-1
        )

        # self.feature_extractor = nn.Conv1d(n_mels, 3, 1, padding=0, bias=False)
        features = self.feature_extractor(spec_diff)
        features = self.activation(features)

        if not self.threshold:
            out = self.final_activation(features)
            return out
        drum_spec = self.drum_threshold(features[:, 0, :].unsqueeze(-2))
        hihat_spec = self.hihat_threshold(features[:, 1, :].unsqueeze(-2))
        snare_spec = self.snare_threshold(features[:, 2, :].unsqueeze(-2))

        # spec_flux = torch.sum(spec_diff, dim=-2, keepdim=True)
        spec_flux = self.onset_threshold(features[:, 3, :].unsqueeze(-2))
        out = torch.cat((drum_spec, hihat_spec, snare_spec, spec_flux), dim=-2)
        out = self.final_activation(out)
        return out


class Threshold(nn.Module):
    def __init__(self, channels=1, mean_range=1, max_range=1, norm_range=1, **kwargs):
        super(Threshold, self).__init__()
        self.threshold = nn.Conv1d(channels, channels, 1, padding=0, bias=True)
        P.register_parametrization(self.threshold, "weight", torch.nn.LeakyReLU())
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
        max_diff = maximums - threshold
        threshold_diff = x - threshold
        x = self.Relu(max_diff) * self.Relu(threshold_diff) * x
        """
        mask = x[..., 1:] > 0
        mask = torch.cat((mask, torch.zeros_like(x[..., 0].bool().unsqueeze(-1))), dim=-1)
        x = x * ~mask
        mask = x[..., 2:] > 0
        mask = torch.cat((mask, torch.zeros_like(x[..., :2].bool())), dim=-1)
        x = x * ~mask
        """

        return x


if __name__ == "__main__":
    spectral_flux = SpecFlux(sample_rate=48000, device="cpu")
