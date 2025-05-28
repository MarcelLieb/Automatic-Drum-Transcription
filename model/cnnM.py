import torch
from torch import nn
from torch.nn import functional as f

from model import ResidualBlock
from model.mamba.mamba import MambaConfig, Mamba


class CNNMamba(nn.Module):
    def __init__(
        self,
        n_mels,
        n_classes,
        d_state,
        d_conv,
        expand,
        flux,
        activation,
        causal,
        num_channels,
        n_layers,
        dropout=0.1,
    ):
        super(CNNMamba, self).__init__()

        self.activation = activation
        self.flux = flux
        self.n_dims = n_mels * (1 + flux)
        self.causal = causal
        self.conv1 = ResidualBlock(
            1,
            num_channels,
            kernel_size=3,
            activation=self.activation,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = ResidualBlock(
            num_channels,
            num_channels,
            kernel_size=3,
            activation=self.activation,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout2 = nn.Dropout(dropout)

        config = MambaConfig(
            d_model=num_channels * (self.n_dims // 4),
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand,
        )
        self.mamba = Mamba(config)

        self.fc1 = nn.Linear(num_channels * (self.n_dims // 4), n_classes)

    def forward(self, x):
        if self.flux:
            diff = x[..., 1:] - x[..., :-1]
            diff = f.relu(f.pad(diff, (1, 0), mode="constant", value=0))
            x = torch.hstack((x, diff))
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = x.reshape(x.size(0), -1, x.size(3))
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        print(x.shape)
        x = self.fc1(x)
        x = x.permute(0, 2, 1)
        x = self.activation(x)
        return x
