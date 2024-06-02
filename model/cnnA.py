import torch
from torch import nn
from torch.nn import functional as f


from model import ResidualBlock1d


class CNNAttention(nn.Module):
    def __init__(
        self, n_mels, n_classes, flux, activation, causal, num_channels, dropout=0.1
    ):
        super(CNNAttention, self).__init__()
        self.activation = activation
        self.flux = flux
        self.n_dims = n_mels * (1 + flux)
        self.causal = causal
        self.conv1 = ResidualBlock1d(
            1,
            num_channels,
            kernel_size=3,
            activation=self.activation,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv2 = ResidualBlock1d(
            num_channels,
            num_channels,
            kernel_size=3,
            activation=self.activation,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.attention = nn.MultiheadAttention(
            embed_dim=num_channels * (self.n_dims // 4),
            num_heads=2,
            dropout=dropout,
            batch_first=True,
        )
        self.fc1 = nn.Linear(num_channels * (self.n_dims // 4), n_classes)

    def forward(self, x):
        if self.flux:
            diff = x[..., 1:] - x[..., :-1]
            diff = f.relu(f.pad(diff, (1, 0), mode="constant", value=0))
            x = torch.hstack((x, diff))
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.reshape(x.size(0), -1, x.size(3))
        x = x.permute(0, 2, 1)
        mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(1), device=x.device
        )
        x, _ = self.attention(
            x, x, x, attn_mask=mask, is_causal=self.causal, need_weights=False
        )
        x = self.fc1(x)
        x = x.permute(0, 2, 1)
        x = self.activation(x)
        return x
