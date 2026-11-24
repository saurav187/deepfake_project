"""Frozen CPU runtime. The student consumes only valid frames from the shared frontend."""

import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from .audio import SR, decode, prepare, windows
from .data import digest
from .models import tensor


class Detector:
    def __init__(self, path):
        info = json.loads((path / "release.json").read_text())
        if info["sha256"] != digest(path / "student.ts"):
            raise ValueError("Student checksum mismatch")
        self.model = torch.jit.load(str(path / "student.ts"), map_location="cpu").eval()
        self.rule = info
        self.device = "cpu"
        self.info = {
            "name": "v3-distilled",
            "version": info["sha256"][:12],
            "experimental": True,
            "description": "Distilled acoustic detector calibrated on grouped development recordings.",
        }

    @torch.inference_mode()
    def analyze(self, data, filename):
        wave = decode(data)
        x, offset = prepare(wave)
        score = []
        segments = []
        for start, w in windows(x):
            logit = float(self.model(tensor(w, True, "cpu")))
            score.append(logit)
            a, b = self.rule["window"]
            segments.append(
                {
                    "start_s": (offset + start) / SR,
                    "end_s": (offset + start + len(w)) / SR,
                    "fake_probability": float(expit(a * logit + b)),
                }
            )
        result = {
            "model": self.info,
            "duration_s": len(wave) / SR,
            "speech_s": len(x) / SR,
            "segments": segments,
            "fake_probability": None,
            "verdict": "insufficient_audio",
            "warnings": [],
        }
        if not score:
            result["warnings"] = ["Less than one second of usable audio."]
            return result
        a, b = self.rule["file"]
        p = float(expit(a * np.mean(score) + b))
        lo, hi = self.rule["thresholds"]
        verdict = (
            "likely_fake" if p >= hi else ("likely_real" if p < lo else "uncertain")
        )
        if len(x) < 4 * SR:
            verdict = "uncertain"
            result["warnings"] = ["Less than four seconds of usable audio."]
        return dict(result, verdict=verdict, fake_probability=p)


def load_detector():
    return Detector(Path(os.environ["V3_RELEASE"]))
