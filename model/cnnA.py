import torch
from torch import nn
from torch.nn import functional as f


from model import PositionalEncoding, AttentionBlock, ResidualBlock


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
    ):
        super(CNNAttention, self).__init__()
        self.activation = activation
        self.flux = flux
        self.n_dims = n_mels * (1 + flux)
        self.causal = causal
        self.conv1 = ResidualBlock(
            1,
            num_channels,
            kernel_size=3,
            activation=self.activation,
            causal=causal,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = ResidualBlock(
            num_channels,
            num_channels,
            kernel_size=3,
            activation=self.activation,
            causal=causal,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout2 = nn.Dropout(dropout)

        self.pos_enc = PositionalEncoding(num_channels * (self.n_dims // 4), dropout)

        self.use_relative_pos = use_relative_pos

        self.attention_blocks = nn.ModuleList()
        for _ in range(num_attention_blocks):
            self.attention_blocks.append(
                AttentionBlock(
                    num_channels * (self.n_dims // 4),
                    num_heads,
                    self.causal,
                    self.activation,
                    dropout,
                    is_causal=self.causal,
                    expansion_factor=expansion_factor,
                    use_relative_pe=use_relative_pos,
                )
            )

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
        if not self.use_relative_pos:
            x = self.pos_enc(x)
        for attention_block in self.attention_blocks:
            x = attention_block(x)
        x = self.fc1(x)
        x = x.permute(0, 2, 1)
        x = self.activation(x)
        return x
