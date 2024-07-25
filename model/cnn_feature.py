import torch
from torch import nn

from model import ResidualBlock


class CNNFeature(nn.Module):
    def __init__(self, num_channels, n_layers, down_sample_factor, channel_multiplication, activation, causal, dropout):
        super(CNNFeature, self).__init__()

        self.activation = activation
        self.down_sample_factor = down_sample_factor

        self.conv_pools = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.conv_pools.append(
                    nn.Sequential(
                        ResidualBlock(
                            1,
                            num_channels * (channel_multiplication ** i),
                            kernel_size=3,
                            causal=causal,
                            activation=self.activation,
                        ),
                        nn.MaxPool2d(
                            kernel_size=(down_sample_factor, 1),
                            stride=(down_sample_factor, 1),
                        ),
                        nn.Dropout(dropout),
                    )
                )
                continue

            self.conv_pools.append(
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
                    nn.Dropout(dropout),
                )
            )

        self.resamples = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.resamples.append(
                    nn.Conv2d(
                        1,
                        num_channels * (channel_multiplication ** i),
                        kernel_size=1,
                        stride=1,
                    )
                )
                continue
            self.resamples.append(
                nn.Conv2d(
                    num_channels * (channel_multiplication ** (i - 1)),
                    num_channels * (channel_multiplication ** i),
                    kernel_size=1,
                    stride=1,
                )
            )

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv_pool, resample in zip(self.conv_pools, self.resamples):
            skip = x
            x = conv_pool(x)
            x = x + resample(torch.nn.functional.interpolate(skip, size=x.shape[2:], mode="nearest"))

        return x

