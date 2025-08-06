import torch
from torch import nn

from model import ResidualBlock


@torch.compile(fullgraph=True, dynamic=True)
class CNNFeature(nn.Module):
    def __init__(
        self, num_channels, n_layers, down_sample_factor, channel_multiplication, activation, causal, dropout,
        in_channels=1
    ):
        super(CNNFeature, self).__init__()

        self.activation = activation
        self.down_sample_factor = down_sample_factor

        conv = []

        for i in range(n_layers):
            if i == 0:
                conv.append(
                    nn.Sequential(
                        ResidualBlock(
                            in_channels,
                            num_channels * (channel_multiplication ** i),
                            kernel_size=3,
                            causal=causal,
                            activation=self.activation,
                        ),
                        nn.MaxPool2d(
                            kernel_size=(down_sample_factor, 1),
                            stride=(down_sample_factor, 1),
                        ),
                        nn.Dropout2d(dropout),
                    )
                )
                continue

            conv.append(
                nn.Sequential(
                    ResidualBlock(
                        num_channels * (channel_multiplication ** (i - 1)),
                        num_channels * (channel_multiplication ** i),
                        kernel_size=3,
                        causal=causal,
                        activation=self.activation,
                    ),
                    nn.MaxPool2d(
                        kernel_size=(down_sample_factor, 1),
                        stride=(down_sample_factor, 1),
                    ),
                    nn.Dropout2d(dropout),
                )
            )

        self.conv_pools = nn.Sequential(*conv) if n_layers > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv_pools(x)

        return x
