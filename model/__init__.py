import math
from copy import deepcopy

import torch
from torch import nn as nn
from torch.nn import functional as f


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        attention_mask,
        causal,
        activation,
        dropout=0.1,
        expansion_factor: int = 4,
        use_relative_pe=True,
    ):
        super(AttentionBlock, self).__init__()
        self.projection = not use_relative_pe
        self.use_relative_pe = use_relative_pe and not causal
        self.multihead = (
            nn.MultiheadAttention(
                embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
            )
            if not self.use_relative_pe
            else MultiHeadSelfAttention(
                L=39_000, d_model=d_model, n_head=n_heads, use_relative_pe=True
            )
        )
        self.activation = activation
        self.causal = causal
        self.mask = attention_mask
        self.dense1 = nn.Linear(d_model, d_model * expansion_factor)
        self.dense2 = nn.Linear(d_model * expansion_factor, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        skip = x
        if self.use_relative_pe:
            x = self.multihead(x)
        else:
            x, _ = (
                self.multihead(
                    x, x, x, attn_mask=self.mask, need_weights=False, is_causal=self.causal
                )
            )
        x = x + skip
        x = self.norm1(x)
        skip = x
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = x + skip
        x = self.norm2(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, L, d_model, n_head, use_relative_pe=True):
        super(MultiHeadSelfAttention, self).__init__()
        self.W_qkv = nn.Linear(d_model, d_model * 3)
        self.W_o = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.n_head = n_head
        self.init_weights()
        self.use_relative_pe = use_relative_pe
        self.L = L

        if use_relative_pe:
            self.position_bias = nn.Parameter(torch.zeros(2 * L - 1, n_head))
            abs_pos = torch.arange(0, L).unsqueeze(0)  # [L] | int
            rel_pos = abs_pos.T - abs_pos  # [L, L] | int | {-(L-1), ..., 0, ..., (L-1)}
            self.rel_pos_idx = rel_pos + (L - 1)  # [L, L] | int | {0, ..., (2*L-1)}

    def init_weights(self):
        nn.init.xavier_uniform_(self.W_qkv.weight)
        nn.init.zeros_(self.W_qkv.bias)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.zeros_(self.W_o.bias)

    def forward(self, x):
        # x_input : [B, L, d_model]
        B, L, d_model = x.shape
        d_head = self.d_model // self.n_head

        qkv = self.W_qkv(x)  # [B, L, d_model*3]
        qkv = qkv.reshape(B, L, self.n_head, d_head * 3)
        qkv = qkv.permute(0, 2, 1, 3)  # [B, n_head, L, d_head*3]
        q, k, v = qkv.chunk(3, dim=-1)  # Each tensor : [B, n_head, L, d_head]

        # Compute the attention scores from q and k, and combine the values in v
        dot_products = torch.matmul(q, k.transpose(-2, -1))  # [B, n_head, L, L]

        if self.use_relative_pe:
            b_pos = self.position_bias[self.rel_pos_idx.view(-1)]  # [L*L, n_head]
            b_pos = b_pos.view(self.L, self.L, self.n_head)
            b_pos = b_pos[:L, :L, :]  # [L, L, n_head]
            b_pos = torch.transpose(
                torch.transpose(b_pos, 1, 2), 0, 1
            )  # [n_head, L, L]
            b_pos = b_pos.unsqueeze(0)  # [B, n_head, L, L]

            dot_products = dot_products + b_pos

        attention = f.softmax(
            dot_products / math.sqrt(d_head),  # d_head = d_k in the paper
            dim=-1,
        )  # [B, n_head, L, L]
        att_vals = torch.matmul(attention, v)  # [B, n_head, L, d_head]
        att_vals = att_vals.permute(0, 2, 1, 3)  # [B, L, n_head, d_head]
        att_vals = att_vals.reshape(B, L, self.d_model)  # [B, L, n_head, d_model]

        # Combine the information from different heads
        att_vals = self.W_o(att_vals)  # [B, L, n_head, d_model]

        return att_vals


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
        # self.norm = nn.BatchNorm1d(out_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        # bs, chan, h, w = x.shape
        # x = x.permute(0, 3, 1, 2)
        # x = x.reshape(bs * w, chan, h)
        x = self.norm(x)
        # x = x.reshape(bs, w, chan, h)
        # x = x.permute(0, 2, 3, 1)
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
