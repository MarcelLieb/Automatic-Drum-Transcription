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
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.causal_conv.padding[0] != 0:
            x = x[..., :-self.causal_conv.padding[0]]
        return x


class CausalMaxPool1d(nn.Module):
    def __init__(self, kernel_size, stride=1, dilation=1, **kwargs):
        super(CausalMaxPool1d, self).__init__()
        self.padding = ((kernel_size - 1) * dilation, 0)
        self.max_pool = nn.MaxPool1d(kernel_size, stride, padding=0, dilation=dilation, **kwargs)

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
