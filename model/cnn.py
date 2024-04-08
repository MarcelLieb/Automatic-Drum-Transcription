from torch import nn
from torch.nn import functional as F
from model.model import CausalConv2d


class CNN(nn.Module):
    def __init__(self, n_mels=82, n_classes=3):
        super(CNN, self).__init__()
        self.n_mels = n_mels
        self.conv1 = CausalConv2d(1, 16, kernel_size=7, bias=False)
        self.norm1 = nn.BatchNorm2d(16)
        self.conv2 = CausalConv2d(16, 32, kernel_size=3, bias=False, dilation=1)
        self.norm2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * n_mels, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.selu(x)
        x = self.conv2(x)
        # x = self.norm2(x)
        x = F.selu(x)
        x = x.reshape((batch_size, 32 * self.n_mels, -1))
        x = x.permute(0, 2, 1)
        x = F.selu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = x.permute(0, 2, 1)
        return x