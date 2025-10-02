from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as f

from model import CausalConv2d
from model.cnn_feature import CNNFeature


class CRNN(nn.Module):
    def __init__(
        self, n_mels, n_classes, num_channels, num_conv_layers, num_rnn_layers, flux, activation, causal,
        down_sample_factor, channel_multiplication, rnn_units, classifier_dim, dropout=None, cnn_dropout=0.3,
        rnn_dropout=0.5, dense_dropout=0.5, use_dense=True,
    ):
        super(CRNN, self).__init__()
        self.activation = activation
        self.causal = causal
        self.n_dims = (n_mels // (down_sample_factor ** num_conv_layers)) * num_channels * (
            channel_multiplication ** (num_conv_layers - 1)) if num_conv_layers > 0 else n_mels * (1 + flux)
        self.flux = flux

        if dropout is not None:
            cnn_dropout = dropout
            rnn_dropout = dropout
            dense_dropout = dropout

        self.conv = CNNFeature(
            num_channels=num_channels,
            n_layers=num_conv_layers,
            down_sample_factor=down_sample_factor,
            channel_multiplication=channel_multiplication,
            # activation=nn.ReLU(),
            activation=activation,
            causal=causal,
            dropout=cnn_dropout,
            in_channels=1 + flux,
        )
        self.rnn = nn.GRU(
            self.n_dims,
            rnn_units,
            num_rnn_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=not causal
        )
        self.fc = nn.Sequential(
            nn.Linear(rnn_units * (1 + (not causal)), classifier_dim),
            activation,
            nn.Dropout(dense_dropout),
            nn.Linear(classifier_dim, n_classes),
        ) if use_dense else nn.Linear(rnn_units * (1 + (not causal)), n_classes)

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

        conv_class = partial(nn.Conv2d, padding=1) if not causal else CausalConv2d
        self.backbone = nn.Sequential(
            conv_class(1, 32, kernel_size=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            conv_class(32, 32, kernel_size=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
            nn.Dropout2d(0.3),
            conv_class(32, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            conv_class(64, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
            nn.Dropout2d(0.3),
        )

        self.rnn = nn.GRU(
            input_size=(n_mels // 9) * 64,
            hidden_size=60,
            num_layers=3,
            batch_first=True,
            dropout=0,
            bidirectional=not causal,
        )

        self.fc = nn.Linear(60, n_classes)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)

        x = self.backbone(x)
        x = x.view(x.size(0), -1, x.size(-1)).permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x