import torch
from torch import nn
from torch.nn import functional as f

from model import PositionalEncoding, AttentionBlock
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
        num_heads=8,
        expansion_factor=4,
        context_size=25,
        use_relative_pos=False,
        dropout=0.1,
        down_sample_factor=3,
        num_conv_layers=2,
        channel_multiplication=2,
    ):
        super(CNNAttention, self).__init__()
        self.activation = activation
        self.flux = flux
        self.n_dims = (n_mels // (down_sample_factor ** num_conv_layers)) * num_channels * (
            channel_multiplication ** (num_conv_layers - 1)) if num_conv_layers > 0 else n_mels * (1 + flux)
        self.causal = causal
        self.backbone = CNNFeature(
            num_channels,
            num_conv_layers,
            down_sample_factor,
            channel_multiplication,
            activation,
            causal,
            dropout,
            in_channels=1 + flux,
        )

        self.pos_enc = PositionalEncoding(self.n_dims, dropout)

        self.use_relative_pos = use_relative_pos

        self.attention_blocks = nn.ModuleList()
        for _ in range(num_attention_blocks):
            self.attention_blocks.append(
                AttentionBlock(
                    self.n_dims,
                    num_heads,
                    self.causal,
                    self.activation,
                    dropout,
                    is_causal=self.causal,
                    expansion_factor=expansion_factor,
                    use_relative_pe=use_relative_pos,
                )
            )

        self.fc1 = nn.Linear(self.n_dims, self.n_dims * expansion_factor)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.n_dims * expansion_factor, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.flux:
            diff = x[..., 1:] - x[..., :-1]
            diff = f.relu(f.pad(diff, (1, 0), mode="constant", value=0))
            x = torch.concatenate((x, diff), dim=1)
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1, x.size(3))
        x = x.permute(0, 2, 1)
        if not self.use_relative_pos:
            x = self.pos_enc(x)
        for attention_block in self.attention_blocks:
            x = attention_block(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x
