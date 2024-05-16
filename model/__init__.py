from copy import deepcopy

import torch
from torch import nn as nn


class ModelEmaV2(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
            **kwargs,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.causal_conv.padding[0] != 0:
            x = x[..., : -self.causal_conv.padding[0]]
        return x


class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size // 2, (kernel_size - 1) * dilation),
            dilation=dilation,
            **kwargs,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.conv.padding[1] != 0:
            x = x[..., : -self.conv.padding[1]]
        return x


class CausalMaxPool1d(nn.Module):
    def __init__(self, kernel_size, stride=1, dilation=1, **kwargs):
        super(CausalMaxPool1d, self).__init__()
        self.padding = ((kernel_size - 1) * dilation, 0)
        self.max_pool = nn.MaxPool1d(
            kernel_size, stride, padding=0, dilation=dilation, **kwargs
        )

    def forward(self, x):
        x = nn.functional.pad(x, self.padding)
        x = self.max_pool(x)
        return x


class CausalAvgPool1d(nn.Module):
    def __init__(self, kernel_size, stride=1, dilation=1, **kwargs):
        super(CausalAvgPool1d, self).__init__()
        self.padding = ((kernel_size - 1) * dilation, 0)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride, padding=0, **kwargs)

    def forward(self, x):
        x = nn.functional.pad(x, self.padding)
        x = self.avg_pool(x)
        return x


class Conv1dVertical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv1dVertical, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            bias=False,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv1dVNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=nn.ELU()):
        super(Conv1dVNormActivation, self).__init__()
        self.conv = Conv1dVertical(in_channels, out_channels, kernel_size)
        # TODO: Ask if the norm is used properly
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=nn.ELU()):
        super(ResidualBlock1d, self).__init__()
        self.conv1 = Conv1dVNormActivation(
            in_channels, out_channels, kernel_size, activation=activation
        )
        self.conv2 = Conv1dVNormActivation(
            out_channels, out_channels, kernel_size, activation=activation
        )
        self.re_sample = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        residual = x
        if self.re_sample is not None:
            residual = self.re_sample(residual)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class Conv2dNormActivationPool(nn.Module):
    def __init__(
        self,
        causal,
        in_channels,
        out_channels,
        kernel_size,
        activation=nn.ELU(),
        pooling=nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        **kwargs,
    ):
        super(Conv2dNormActivationPool, self).__init__()
        self.conv = (
            CausalConv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
            if causal
            else nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                bias=False,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2)
                if isinstance(kernel_size, tuple)
                else kernel_size // 2,
                **kwargs,
            )
        )
        self.activation = activation
        self.norm = nn.BatchNorm2d(out_channels)
        self.pooling = pooling

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        if self.pooling is not None:
            x = self.pooling(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        causal=True,
        activation=nn.ELU(),
        **kwargs,
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = (
            CausalConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                bias=False,
                **kwargs,
            )
            if causal
            else nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                bias=False,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2)
                if isinstance(kernel_size, tuple)
                else kernel_size // 2,
                **kwargs,
            )
        )
        self.conv2 = (
            CausalConv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                bias=False,
                **kwargs,
            )
            if causal
            else nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                bias=False,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2)
                if isinstance(kernel_size, tuple)
                else kernel_size // 2,
                **kwargs,
            )
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = activation
        self.re_sample = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        residual = x
        if self.re_sample is not None:
            residual = self.re_sample(residual)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x + residual
