"""Model A: frozen self-supervised speech encoder + learned layer mix + small head."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

from detector import paths

ENCODERS = {
    "wavlm-base-plus": paths.DETECTOR / "pretrained/wavlm-base-plus",
    "wav2vec2-xls-r-300m": paths.DETECTOR / "pretrained/wav2vec2-xls-r-300m",
}


class AttentiveStatsPool(nn.Module):
    """Attention-weighted mean and standard deviation over time."""
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):                                   # x: (B, T, D)
        w = torch.softmax(self.attention(x), dim=1)          # (B, T, 1)
        mean = (w * x).sum(1)
        var = (w * (x - mean.unsqueeze(1)) ** 2).sum(1)
        return torch.cat([mean, torch.sqrt(var.clamp_min(1e-6))], dim=1)


class SSLDetector(nn.Module):
    def __init__(self, encoder: str, proj_dim=128, dropout=0.2):
        super().__init__()
        path = Path(ENCODERS[encoder])
        self.encoder_name = encoder
        self.encoder = AutoModel.from_pretrained(path)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        pre = json.loads((path / "preprocessor_config.json").read_text())
        self.normalize = bool(pre.get("do_normalize", False))
        n_layers = self.encoder.config.num_hidden_layers + 1
        dim = self.encoder.config.hidden_size
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Linear(dim, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.pool = AttentiveStatsPool(proj_dim)
        self.head = nn.Sequential(nn.Linear(2 * proj_dim, proj_dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(proj_dim, 1))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # frozen: keep dropout/feature-extractor behaviour fixed
        return self

    def trainable_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("encoder.")}

    def load_trainable_state(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith("encoder.") for k in missing), (missing, unexpected)

    def forward(self, wave):                                # wave: (B, samples) at 16 kHz
        if self.normalize:
            wave = (wave - wave.mean(1, keepdim=True)) / (wave.std(1, keepdim=True) + 1e-7)
        with torch.no_grad():
            hidden = self.encoder(wave, output_hidden_states=True).hidden_states
        stack = torch.stack(hidden, dim=0).float()          # (L, B, T, D)
        weights = torch.softmax(self.layer_logits, 0).view(-1, 1, 1, 1)
        x = self.norm((weights * stack).sum(0))
        return self.head(self.pool(self.proj(x))).squeeze(1)  # logits (B,)
