import torch
from torch import nn

from model import ResidualBlock


class CNNFeature(nn.Module):
    def __init__(self, num_channels, down_sample_factor, activation, causal):
        super(CNNFeature, self).__init__()

        self.activation = activation
        self.down_sample_factor = down_sample_factor


        self.conv1 = ResidualBlock(
            1, num_channels, kernel_size=3, causal=causal, activation=self.activation
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(down_sample_factor, 1), stride=(down_sample_factor, 1)
        )

        self.conv2 = ResidualBlock(
            num_channels,
            num_channels * 2,
            kernel_size=3,
            causal=causal,
            activation=self.activation,
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size=(down_sample_factor, 1), stride=(down_sample_factor, 1)
        )

        self.resample = nn.Conv2d(
            num_channels, num_channels * 2, kernel_size=1, stride=1
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool1(x)
        skip = x
        x = self.conv2(x)
        x = self.pool2(x)
        x = x + self.resample(
            torch.nn.functional.interpolate(skip, size=x.shape[2:], mode="nearest")
        )

        return x

