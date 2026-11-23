"""Released detector: the app backend (PRD section 7) and the engine of the sealed evaluation.

The app loads it with DETECTOR_BACKEND=detector, which calls load_detector().
Environment: DETECTOR_RELEASE (default detector/runs/release/v1), DEVICE (auto/cpu/cuda).

Pipeline per recording, identical for evaluation and serving:
  bytes -> decode (16 kHz mono) -> prepare (trim, edge cut and fade, loudness) ->
  4 s windows on the shared schedule (audio.window_starts, every window) ->
  frozen encoder once per window -> each member head -> mean logit over members ->
  recording logit = mean over windows -> Platt -> verdict from the frozen thresholds.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel

from detector import paths
from detector.audio import SR, decode, prepare_with_offset, window_starts
from detector.model.ssl import ENCODERS, AttentiveStatsPool

WINDOW = 4 * SR
BATCH = 16


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class Head(nn.Module):
    """The trainable part of model.ssl.SSLDetector, with identical parameter names,
    so a saved trainable state loads directly."""
    def __init__(self, n_layers, dim, proj_dim=128, dropout=0.2):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Linear(dim, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.pool = AttentiveStatsPool(proj_dim)
        self.head = nn.Sequential(nn.Linear(2 * proj_dim, proj_dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(proj_dim, 1))

    def forward(self, stack):                                   # stack: (L, B, T, D) float
        weights = torch.softmax(self.layer_logits, 0).view(-1, 1, 1, 1)
        x = self.norm((weights * stack).sum(0))
        return self.head(self.pool(self.proj(x))).squeeze(1)


def windows_of(wave):
    """Every 4 s window of a prepared recording on the shared schedule (short ones repeated)."""
    if len(wave) < WINDOW:
        wave = np.tile(wave, int(np.ceil(WINDOW / max(len(wave), 1))))[:WINDOW]
    starts = window_starts(len(wave), WINDOW)
    return np.stack([wave[s:s + WINDOW] for s in starts]), starts


class Detector:
    def __init__(self, release_dir=None, device=None):
        release_dir = Path(release_dir or os.getenv("DETECTOR_RELEASE", paths.RUNS / "release" / "v1"))
        if not release_dir.is_absolute():
            release_dir = paths.REPO / release_dir
        release_file = release_dir / "release.json"
        expected = (release_dir / "release.sha256").read_text().split()[0]
        if sha256(release_file) != expected:
            raise ValueError(f"{release_file} does not match release.sha256")
        self.release = json.loads(release_file.read_text())
        self.release_sha256 = expected
        device = device or os.getenv("DEVICE", "auto")
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else ("cpu" if device == "auto" else device))
        encoder_path = ENCODERS[self.release["encoder"]]
        self.encoder = AutoModel.from_pretrained(encoder_path).to(self.device).eval()
        pre = json.loads((Path(encoder_path) / "preprocessor_config.json").read_text())
        self.normalize = bool(pre.get("do_normalize", False))
        n_layers = self.encoder.config.num_hidden_layers + 1
        self.heads = []
        for member in self.release["members"]:
            ckpt = Path(member["run"])
            ckpt = (ckpt if ckpt.is_absolute() else paths.REPO / ckpt) / "best.pt"
            if sha256(ckpt) != member["checkpoint_sha256"]:
                raise ValueError(f"{ckpt} does not match the release")
            saved = torch.load(ckpt, map_location="cpu", weights_only=True)
            if saved["encoder"] != self.release["encoder"]:
                raise ValueError(f"{ckpt} was trained with another encoder")
            head = Head(n_layers, self.encoder.config.hidden_size)
            head.load_state_dict(saved["state"])
            self.heads.append(head.to(self.device).eval())
        cal, thr = self.release["calibration"], self.release["thresholds"]
        self.a, self.b = cal["a"], cal["b"]
        self.t_real, self.t_fake = thr["likely_real_below"], thr["likely_fake_at_or_above"]
        self.min_speech_s = thr["min_speech_s"]
        self.info = {
            "name": f"ssl-{self.release['encoder']}",
            "version": f"{self.release['name']}-{expected[:8]}",
            "experimental": True,
            "description": (f"Frozen {self.release['encoder']} encoder with {len(self.heads)} trained "
                            f"head{'s' if len(self.heads) != 1 else ''}; "
                            "calibrated on ASVspoof 5 and IndieFake dev audio (50/50 prior). Segment colours "
                            "show relative evidence per 4 s window, not calibrated probabilities."),
        }

    def probability(self, logit):
        return 1.0 / (1.0 + math.exp(-(self.a * logit + self.b)))

    @torch.inference_mode()
    def window_logits(self, windows: np.ndarray) -> np.ndarray:
        """Mean logit over member heads for each (n, 64000) window."""
        out = []
        for i in range(0, len(windows), BATCH):
            x = torch.from_numpy(np.ascontiguousarray(windows[i:i + BATCH])).to(self.device)
            if self.normalize:
                x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                stack = torch.stack(self.encoder(x, output_hidden_states=True).hidden_states, 0).float()
                logits = torch.stack([h(stack).float() for h in self.heads]).mean(0)
            out.append(logits.cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)

    def verdict(self, p):
        if p >= self.t_fake:
            return "likely_fake"
        return "likely_real" if p < self.t_real else "uncertain"

    def analyze_wave(self, wave: np.ndarray) -> dict:
        prepared, offset = prepare_with_offset(wave)
        speech_s = len(prepared) / SR
        duration_s = len(wave) / SR
        base = dict(duration_s=duration_s, speech_s=speech_s, model=self.info)
        if speech_s < self.min_speech_s:
            return dict(base, verdict="insufficient_audio", fake_probability=None, segments=[],
                        warnings=["Less than 1 second of clear speech was found."])
        windows, starts = windows_of(prepared)
        logits = self.window_logits(windows)
        recording = float(np.mean(logits))
        p = self.probability(recording)
        segments = [dict(start_s=(offset + s) / SR, end_s=min(offset + s + WINDOW, offset + len(prepared)) / SR,
                         fake_probability=self.probability(float(l))) for s, l in zip(starts, logits)]
        warnings = []
        if speech_s < 4:
            warnings.append("Short recording: the result is less reliable.")
        return dict(base, verdict=self.verdict(p), fake_probability=p, segments=segments, warnings=warnings,
                    recording_logit=recording)

    def analyze(self, data: bytes, filename: str) -> dict:
        """PRD section 7 entry point. Raises ValueError for undecodable input."""
        try:
            wave, _ = decode(data)
        except Exception as exc:
            raise ValueError("Unreadable audio") from exc
        if len(wave) == 0:
            raise ValueError("Unreadable audio")
        result = self.analyze_wave(wave)
        result.pop("recording_logit", None)
        return result


def load_detector():
    """Called by the app when DETECTOR_BACKEND=detector."""
    return Detector()
