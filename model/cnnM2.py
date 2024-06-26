import torch
from mamba_ssm import Mamba
from torch import nn
from torch.nn import functional as f

from model import ResidualBlock

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm = None

# RMSNorm doesn't work with 1080Ti
if torch.cuda.get_device_capability(0)[0] <= 6:
    RMSNorm = None


class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=f.silu,
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (
                (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        )
        self.fc1 = nn.Linear(
            in_features, 2 * hidden_features, bias=bias, **factory_kwargs
        )
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y


class MambaBlock(nn.Module):
    def __init__(
            self,
            d_model,
            d_state,
            d_conv,
            expand,
            d_intermediate=None,
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
        self.norm2 = nn.LayerNorm(d_model) if RMSNorm is None else RMSNorm(d_model)
        self.mlp = (
            None
            if d_intermediate is None
            else GatedMLP(d_model, hidden_features=None, out_features=d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.mamba(x)
        x = self.norm(x + residual)
        if self.mlp is not None:
            residual = x
            x = self.mlp(x)
            x = self.norm2(x + residual)
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
            dropout=0.1,
    ):
        super(CNNMambaFast, self).__init__()

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
        mamba_layers = [
            MambaBlock(
                d_model=num_channels * (self.n_dims // 4),
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ]
        self.mamba = nn.Sequential(*mamba_layers)
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
        x = self.fc1(x)
        x = x.permute(0, 2, 1)
        x = self.activation(x)
        return x
