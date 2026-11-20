import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) if pool else nn.Identity(),
        )
    def forward(self, x):
        return self.layers(x)


class FixedAdaptivePool(nn.Module):
    """The original 16×11 to 8×8 adaptive average, with deterministic CUDA backward."""
    def __init__(self):
        super().__init__()
        weights = torch.zeros(11, 8)
        for col in range(8):
            start = col * 11 // 8
            end = ((col + 1) * 11 + 7) // 8
            weights[start:end, col] = 1 / (end - start)
        self.register_buffer('weights', weights, persistent=False)

    def forward(self, x):
        return x.unflatten(2, (8, 2)).mean(3) @ self.weights


class CNNLSTMDetector(nn.Module):
    def __init__(self, lstm_hidden: int = 256, lstm_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.block1 = ConvBlock(1,   32, pool=True)
        self.block2 = ConvBlock(32,  64, pool=True)
        self.block3 = ConvBlock(64, 128, pool=True)
        self.ap     = FixedAdaptivePool()
        lstm_input_size = 128 * 8
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = self.ap(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, H, C * W)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(h)
