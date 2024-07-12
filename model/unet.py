from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ResidualBlock1d, Conv1dVNormActivation


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.SELU()):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            Conv1dVNormActivation(in_channels, out_channels, kernel_size=3, activation=activation),
            Conv1dVNormActivation(out_channels, out_channels, kernel_size=3, activation=activation),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.max_pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 1), stride=(2, 1))
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, channels: int, checkpoint: Optional[str] = None):
        super(UNet, self).__init__()

        self.inc = DoubleConv(1, channels)

        self.down1 = Down(channels, channels * 2)
        self.down2 = Down(channels * 2, channels * 4)
        self.down3 = Down(channels * 4, channels * 8)
        self.down4 = Down(channels * 8, channels * 16)

        self.ups = nn.ModuleList()

        for i in range(4):
            self.ups.append(Up(channels * (16 // 2 ** i), channels * (8 // 2 ** i)))

        self.out = nn.Conv2d(channels, 1, kernel_size=1)

        if checkpoint is not None:
            checkpoint = torch.load(checkpoint)
            if checkpoint["model_settings"]["channels"] == channels:
                self.load_state_dict(checkpoint["model"])

    def forward(self, x, return_features: Optional[int] = None):
        x = x.unsqueeze(1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        skips = [x1, x2, x3, x4]

        for i in range(4):
            bs, c, h, w = x.size()
            # print(skips[3 - i].size(), x.size())
            x = self.ups[i](x, skips[3 - i])
            assert x.size() == (bs, c // 2, h * 2, w), f"{x.size()} != {(bs, c // 2, h * 2, w)}"
            if return_features == i:
                return x

        x = self.out(x)
        return x
