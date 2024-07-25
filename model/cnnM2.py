from typing import Literal

import torch
from mamba_ssm import Mamba
from torch import nn
from torch.nn import functional as f

from model import ResidualBlock
from model.unet import UNet
from model.cnn_feature import CNNFeature

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm = None

# RMSNorm doesn't work with 1080Ti
if torch.cuda.get_device_capability(0)[0] <= 6:
    RMSNorm = None


class MambaBlock(nn.Module):
    def __init__(
            self,
            d_model,
            d_state,
            d_conv,
            expand,
            dropout=0.1,
    ):
        super(MambaBlock, self).__init__()

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(d_model) if RMSNorm is None else RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.mamba(x)
        x = self.norm(x + residual)
        x = self.dropout(x)
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
            return_features=1,
            dropout=0.1,
    ):
        super(CNNMambaFast, self).__init__()

        self.activation = activation
        self.flux = flux
        self.n_dims = (n_mels * (1 + flux)) // 4 * 2
        self.causal = causal

        match backbone:
            case "cnn":
                self.backbone = CNNFeature(
                    num_channels=num_channels,
                    n_layers=2,
                    down_sample_factor=2,
                    channel_multiplication=2,
                    activation=activation,
                    causal=causal,
                    dropout=dropout,
                )
            case "unet":
                self.backbone = nn.Sequential(
                    UNet(num_channels // 4, return_features=return_features),
                    ResidualBlock(
                        in_channels=num_channels * (2 // 2 ** return_features),
                        out_channels=num_channels * (2 // 2 ** return_features),
                        kernel_size=3,
                        causal=causal,
                    ),
                )
        self.return_features = return_features if backbone == "unet" else -1
        mamba_layers = [
            MambaBlock(
                d_model=(num_channels * self.n_dims),
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ]
        self.mamba = nn.Sequential(*mamba_layers)
        self.fc1 = nn.Linear(num_channels * self.n_dims, n_classes)

    def forward(self, x):
        if self.flux:
            diff = x[..., 1:] - x[..., :-1]
            diff = f.relu(f.pad(diff, (1, 0), mode="constant", value=0))
            x = torch.hstack((x, diff))
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1, x.size(3))
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = self.fc1(x)
        x = x.permute(0, 2, 1)
        x = self.activation(x)
        return x
