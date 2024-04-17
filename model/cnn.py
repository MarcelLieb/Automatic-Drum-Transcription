from torch import nn
from torch.nn import functional as f
from model import Conv2dNormActivationPool


class CNN(nn.Module):
    def __init__(self, n_mels=82, n_classes=3, num_channels=20, dropout=0.1, ):
        super(CNN, self).__init__()
        self.n_mels = n_mels
        self.conv1 = Conv2dNormActivationPool(causal=True, in_channels=1, out_channels=num_channels, kernel_size=3)
        self.conv2 = Conv2dNormActivationPool(causal=True, in_channels=num_channels, out_channels=num_channels, kernel_size=3)
        self.conv3 = Conv2dNormActivationPool(causal=True, in_channels=num_channels, out_channels=num_channels, kernel_size=3, pooling=None)

        self.fc1 = nn.Linear(num_channels * (n_mels // 4), 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, n_classes)
        self.num_channels = num_channels

    def forward(self, x):
        batch_size = x.size(0)
        # Add channel dimension
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape((batch_size, self.num_channels * (self.n_mels // 4), -1))
        x = x.permute(0, 2, 1)
        x = f.selu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = f.tanh(x)
        x = x.permute(0, 2, 1)
        return x
