import torch
from torch import nn
from torch.nn import functional as f
from torch.nn.attention.flex_attention import create_block_mask

from model import PositionalEncoding, AttentionBlock, causal_fn as causal_fn
from model.cnn_feature import CNNFeature


class CNNAttention(nn.Module):
    def __init__(
        self,
        n_mels,
        n_classes,
        flux,
        activation,
        causal,
        num_channels,
        num_attention_blocks=3,
        hidden_units=128,
        num_heads=8,
        expansion_factor=4,
        use_relative_pos=False,
        dropout=None,
        cnn_dropout=0.3,
        attention_dropout=0.1,
        positional_encoding_dropout: float = 0.1,
        down_sample_factor=3,
        num_conv_layers=2,
        channel_multiplication=2,
        **_kwargs,
    ):
        super(CNNAttention, self).__init__()
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

        if dropout is not None:
            cnn_dropout = dropout
            attention_dropout = dropout
            positional_encoding_dropout = dropout

        self.backbone = CNNFeature(
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

        self.proj = nn.Linear(self.n_dims, hidden_units)

        self.pos_enc = (
            PositionalEncoding(hidden_units, positional_encoding_dropout)
            if not use_relative_pos
            else nn.Identity()
        )

        self.use_relative_pos = use_relative_pos

        self.attention_blocks = nn.ModuleList()
        for _ in range(num_attention_blocks):
            self.attention_blocks.append(
                AttentionBlock(
                    hidden_units,
                    num_heads,
                    causal,
                    self.activation,
                    attention_dropout,
                    is_causal=self.causal,
                    expansion_factor=expansion_factor,
                    use_relative_pe=use_relative_pos,
                )
            )
        self.norm = nn.LayerNorm(hidden_units)
        self.fc = nn.Linear(hidden_units, n_classes)

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
        B, T, C = x.shape
        block_mask = None
        x = self.pos_enc(x)
        if self.causal:
            block_mask = create_block_mask(causal_fn, None, None, T, T, device=x.device)

        for attention_block in self.attention_blocks:
            x = attention_block(x, attn_mask=block_mask)
        x = self.norm(x)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x
