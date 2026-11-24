import numpy as np
import torch
from torch import nn

from .audio import features


class Student(nn.Module):
    """Frame MLP with local context, utterance mean and standard deviation."""

    def __init__(self):
        super().__init__()
        self.frame = nn.Sequential(
            nn.Linear(64 * 3, 256), nn.ReLU(), nn.Linear(256, 192), nn.ReLU()
        )
        self.head = nn.Sequential(nn.Linear(384, 96), nn.ReLU(), nn.Linear(96, 1))

    def forward(self, x):
        left = torch.cat([x[:1], x[:-1]])
        right = torch.cat([x[1:], x[-1:]])
        h = self.frame(torch.cat([left, x, right], dim=1))
        pooled = torch.cat([h.mean(0), h.std(0, unbiased=False)])
        return self.head(pooled.unsqueeze(0)).reshape(())


class Teacher(nn.Module):
    def __init__(self, path):
        super().__init__()
        from transformers import AutoModel

        self.ssl = AutoModel.from_pretrained(str(path), local_files_only=True)
        self.ssl.requires_grad_(False)
        self.ssl.encoder.layers[-2:].requires_grad_(True)
        self.head = nn.Linear(2 * self.ssl.config.hidden_size, 1)

    def forward(self, x):
        h = self.ssl(x[None, :]).last_hidden_state[0]
        return self.head(torch.cat([h.mean(0), h.std(0, unbiased=False)])).squeeze(-1)


def tensor(x, student, device):
    return torch.from_numpy(np.ascontiguousarray(features(x) if student else x)).to(
        device
    )
