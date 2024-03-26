import torchaudio
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, n_mels=82, n_classes=3):
        super(CNN, self).__init__()
        self.n_mels = n_mels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.fc1 = nn.Linear(16 * n_mels, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.reshape((batch_size, 16 * self.n_mels, -1))
        x = x.permute(0, 2, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.tanh(x)
        x = x.permute(0, 2, 1)
        return x