import torch
from torch import nn
from torch.nn import functional as f

from model import ResidualBlock1d
from model.cnn_feature import CNNFeature


class CNN(nn.Module):
    def __init__(
        self,
        n_mels,
        n_classes,
        num_channels,
        dropout,
        num_feature_layers,
        channel_multiplication,
        num_residual_blocks,
        causal,
        flux,
        down_sample_factor: int,
        activation=nn.SELU(),
        classifier_dim=2 ** 6,
    ):
        super(CNN, self).__init__()
        self.activation = activation
        self.flux = flux
        self.n_dims = n_mels
        self.classifier_dim = classifier_dim
        self.down_sample_factor = down_sample_factor

        self.backbone = CNNFeature(
            num_channels=num_channels,
            n_layers=num_feature_layers,
            down_sample_factor=down_sample_factor,
            channel_multiplication=channel_multiplication,
            activation=activation,
            causal=causal,
            dropout=dropout,
            in_channels=1 + flux,
        )

        self.residuals = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residuals.append(
                ResidualBlock1d(
                    num_channels * (channel_multiplication ** (num_feature_layers - 1)),
                    num_channels * (channel_multiplication ** (num_feature_layers - 1)),
                    kernel_size=3,
                    activation=self.activation,
                )
            )

        self.fc1 = nn.Linear(
            num_channels * (channel_multiplication ** (num_feature_layers - 1)) * (
                self.n_dims // (down_sample_factor ** num_feature_layers)), classifier_dim
        )
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(classifier_dim, n_classes)
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        if self.flux:
            diff = x[..., 1:] - x[..., :-1]
            diff = f.relu(f.pad(diff, (1, 0), mode="constant", value=0))
            x = torch.concatenate((x, diff), dim=1)
        x = self.backbone(x)
        for residual in self.residuals:
            x = residual(x)
        bs, ch, h, w = x.size()
        x = x.reshape(bs, ch * h, w)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x
