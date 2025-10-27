from typing import Literal

import torch
from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams
from torch import nn
from torch.nn import functional as f

from model.cnn_feature import CNNFeature


class DenseEncoder(nn.Module):
    def __init__(self, n_mels, num_channels, activation, dropout, flux):
        super(DenseEncoder, self).__init__()
        self.activation = activation
        self.n_dims = n_mels * (1 + flux)
        self.fc1 = nn.Linear(self.n_dims, num_channels)
        self.fc2 = nn.Linear(num_channels, self.n_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = x.reshape(x.size(0), -1, x.size(3))
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        # invert the permutation
        x = x.permute(0, 2, 1)
        x = x.reshape(x.size(0), 1, x.size(1), x.size(2))
        return x


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        d_conv,
        expand,
        layer_idx=0,
        dropout=0.1,
        activation=nn.SiLU(),
        mlp: bool = False,
    ):
        super(MambaBlock, self).__init__()

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, inference_params: InferenceParams | None = None):
        residual = x
        x = self.norm(x)
        x = self.mamba(x, inference_params=inference_params)
        x = self.dropout(x)  # Dropout may lead to nan values
        x = x + residual
        return x


class CNNMambaFast(nn.Module):
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
        backbone: Literal["unet", "cnn"],
        dropout=None,
        cnn_dropout=0.3,
        mamba_dropout=0.5,
        dense_dropout=0.5,
        down_sample_factor: int = 3,
        num_conv_layers: int = 2,
        channel_multiplication: int = 2,
        classifier_dim: int = 128,
        hidden_units: int = 512,
    ):
        super(CNNMambaFast, self).__init__()

        self.activation = activation
        self.flux = flux
        self.n_dims = (
            (n_mels // (down_sample_factor**num_conv_layers))
            * num_channels
            * (channel_multiplication ** (num_conv_layers - 1))
            if num_conv_layers > 0
            else n_mels * (1 + flux)
        )
        self.causal = causal

        # Backward compatibility
        if dropout is not None:
            cnn_dropout = dropout
            mamba_dropout = dropout
            dense_dropout = dropout

        match backbone:
            case "cnn":
                self.backbone = CNNFeature(
                    num_channels=num_channels,
                    n_layers=num_conv_layers,
                    down_sample_factor=down_sample_factor,
                    channel_multiplication=channel_multiplication,
                    activation=activation,
                    causal=causal,
                    dropout=cnn_dropout,
                    in_channels=1 + flux,
                )
            case "dense":
                self.backbone = DenseEncoder(
                    n_mels, num_channels, activation, dropout, flux
                )
        self.proj = nn.Linear(self.n_dims, hidden_units)
        mamba_layers = [
            MambaBlock(
                d_model=hidden_units,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=mamba_dropout,
                layer_idx=i,
                activation=activation,
            )
            for i in range(n_layers)
        ]
        self.mamba = nn.ModuleList(mamba_layers)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_units),
            nn.Linear(hidden_units, classifier_dim),
            activation,
            nn.Dropout(dense_dropout),
            nn.Linear(classifier_dim, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.flux:
            diff = x[..., 1:] - x[..., :-1]
            diff = f.relu(f.pad(diff, (1, 0), mode="constant", value=0))
            x = torch.concatenate((x, diff), dim=1)
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1, x.size(3))
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        for mamba in self.mamba:
            x = mamba(x)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x

    def recurrent(self, x):
        x = x.unsqueeze(1)
        if self.flux:
            diff = x[..., 1:] - x[..., :-1]
            diff = f.relu(f.pad(diff, (1, 0), mode="constant", value=0))
            x = torch.concatenate((x, diff), dim=1)
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1, x.size(3))
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        bs, L, d = x.shape
        infer_params = InferenceParams(max_batch_size=bs, max_seqlen=L)
        bs, L, d = x.shape
        outs = []
        for i in range(L):
            step = x[:, i : i + 1, :]
            for mamba in self.mamba:
                step = mamba(step, inference_params=infer_params)
            infer_params.seqlen_offset += 1
            outs.append(step)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x
