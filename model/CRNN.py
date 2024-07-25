import torch
import torch.nn as nn
import torch.nn.functional as f

from model.cnn_feature import CNNFeature


class CRNN(nn.Module):
    def __init__(self, n_mels, n_classes, num_channels, num_layers, flux, activation, causal, dropout):
        super(CRNN, self).__init__()
        self.activation = activation
        self.causal = causal
        self.n_dims = n_mels * (1 + flux)
        self.flux = flux

        self.conv = CNNFeature(
            num_channels,
            num_conv_layers,
            2,
            1,
            activation,
            causal,
            dropout
        )
        self.rnn = nn.GRU(
            num_channels * self.n_dims // 4,
            num_channels * self.n_dims // 4,
            1,
            batch_first=True,
            dropout=dropout,
            bidirectional=not causal
        )
        self.fc = nn.Linear(num_channels * self.n_dims // 4 * (2 - causal) , n_classes)

    def forward(self, x):
        if self.flux:
            diff = x[..., 1:] - x[..., :-1]
            diff = f.relu(f.pad(diff, (1, 0), mode="constant", value=0))
            x = torch.hstack((x, diff))
        x = self.conv(x)
        x = x.view(x.size(0), -1, x.size(-1)).permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.activation(x)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x
