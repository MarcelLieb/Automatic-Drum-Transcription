import torch
from torch import nn
from torch.nn import functional as f
from model import ResidualBlock, ResidualBlock1d


class CNN(nn.Module):
    def __init__(
        self,
        n_mels,
        n_classes,
        num_channels,
        dropout,
        num_residual_blocks,
        causal,
        flux,
    ):
        super(CNN, self).__init__()
        self.flux = flux
        self.n_dims = n_mels * (1 + flux)

        self.conv1 = ResidualBlock(
            1, num_channels, kernel_size=3, causal=causal, activation=nn.ELU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.conv2 = ResidualBlock(
            num_channels,
            num_channels * 2,
            kernel_size=3,
            causal=causal,
            activation=nn.ELU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.residuals = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residuals.append(
                ResidualBlock1d(
                    num_channels * 2,
                    num_channels * 2,
                    kernel_size=3,
                )
            )

        self.fc1 = nn.Linear(num_channels * 2 * (self.n_dims // 9), 2**5)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(2**5, n_classes)
        self.num_channels = num_channels

    def forward(self, x):
        if self.flux:
            diff = x[1:] - x[-1]
            diff = f.relu(f.pad(diff, (0, 0, 0, 0, 1, 0), mode="constant", value=0))
            x = torch.hstack((x, diff))
        batch_size = x.size(0)
        # Add channel dimension
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        for residual in self.residuals:
            x = residual(x)
            x = self.dropout(x)
        x = x.reshape((batch_size, self.num_channels * 2 * (self.n_dims // 9), -1))
        x = x.permute(0, 2, 1)
        x = f.selu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = f.selu(x)
        # x = f.tanh(x)
        # x = f.sigmoid(x)
        x = x.permute(0, 2, 1)
        return x
