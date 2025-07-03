import torch
import torch.nn as nn
import torch.nn.functional as f

from model.cnn_feature import CNNFeature


class CRNN(nn.Module):
    def __init__(
        self, n_mels, n_classes, num_channels, num_conv_layers, num_rnn_layers, flux, activation, causal,
        dropout, down_sample_factor, channel_multiplication, rnn_units, classifier_dim
    ):
        super(CRNN, self).__init__()
        self.activation = activation
        self.causal = causal
        self.n_dims = (n_mels // (down_sample_factor ** num_conv_layers)) * num_channels * (
            channel_multiplication ** (num_conv_layers - 1)) if num_conv_layers > 0 else n_mels * (1 + flux)
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
            rnn_units,
            num_rnn_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=not causal
        )
        self.fc = nn.Sequential(
            nn.Linear(rnn_units * (1 + (not causal)), classifier_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, n_classes),
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


class CRNN_Vogl(nn.Module):
    def __init__(self, n_mels, n_classes, causal, *args, **kwargs):
        super(CRNN_Vogl, self).__init__()
        self.backbone = CNNFeature(
            32,
            2,
            3,
            2,
            nn.ReLU(),
            causal,
            dropout=0.,
            in_channels=1,
        )

        self.rnn1 = nn.GRU(
            input_size=(n_mels // 9) * 64,
            hidden_size=60,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=not causal,
        )
        self.rnn2 = nn.GRU(
            input_size=60 * (1 + (not causal)),
            hidden_size=60,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        self.fc = nn.GRU(
            input_size=60 * (1 + (not causal)),
            hidden_size=n_classes,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)

        x = self.backbone(x)
        x = x.view(x.size(0), -1, x.size(-1)).permute(0, 2, 1)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x, _ = self.fc(x)
        x = x.permute(0, 2, 1)
        return x