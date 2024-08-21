import torch
import torch.nn as nn
import torch.nn.functional as f

from model.cnn_feature import CNNFeature


class CRNN(nn.Module):
    def __init__(self, n_mels, n_classes, num_channels, num_conv_layers, num_rnn_layers, flux, activation, causal,
                 dropout, down_sample_factor, channel_multiplication, classifier_dim):
        super(CRNN, self).__init__()
        self.activation = activation
        self.causal = causal
        self.n_dims = (n_mels // (down_sample_factor ** num_conv_layers)) * num_channels * (
                channel_multiplication ** (num_conv_layers - 1))
        self.flux = flux

        self.conv = CNNFeature(
            num_channels,
            num_conv_layers,
            down_sample_factor,
            channel_multiplication,
            activation,
            causal,
            dropout,
            in_channels=1 + flux,
        )
        self.rnn = nn.GRU(
            self.n_dims,
            self.n_dims,
            num_rnn_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=not causal
        )
        self.fc = nn.Sequential(
            nn.Linear(self.n_dims * (2 - causal), classifier_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, n_classes),
            activation,
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        if self.flux:
            diff = x[..., 1:] - x[..., :-1]
            diff = f.relu(f.pad(diff, (1, 0), mode="constant", value=0))
            x = torch.concatenate((x, diff), dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1, x.size(-1)).permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.activation(x)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x
