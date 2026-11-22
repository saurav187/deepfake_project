"""Label-neutral training augmentation: real-world codecs, gain and noise.

Lesson from the legacy pipeline: an 8 kHz phone channel alone flipped most
genuine recordings to FAKE. Uploaded audio is usually compressed (WhatsApp
Opus, MP3, AAC/M4A), so the model must see every codec on both classes.
"""
from __future__ import annotations

import subprocess

import numpy as np

from detector.audio import SR

# (name, ffmpeg encoder args, container, decode rate). Bitrates span voice-note to music quality.
CODECS = [
    ("mp3_32k", ["-c:a", "libmp3lame", "-b:a", "32k"], "mp3"),
    ("mp3_64k", ["-c:a", "libmp3lame", "-b:a", "64k"], "mp3"),
    ("mp3_128k", ["-c:a", "libmp3lame", "-b:a", "128k"], "mp3"),
    ("aac_48k", ["-c:a", "aac", "-b:a", "48k"], "adts"),
    ("aac_96k", ["-c:a", "aac", "-b:a", "96k"], "adts"),
    ("opus_16k", ["-c:a", "libopus", "-b:a", "16k", "-application", "voip"], "ogg"),
    ("opus_32k", ["-c:a", "libopus", "-b:a", "32k"], "ogg"),
    ("mulaw_8k", ["-ar", "8000", "-c:a", "pcm_mulaw"], "wav"),
    ("pcm_8k", ["-ar", "8000", "-c:a", "pcm_s16le"], "wav"),
]


def _run(args, data):
    return subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *args], input=data,
                          capture_output=True, check=True).stdout


def apply_codec(wave: np.ndarray, name: str) -> np.ndarray:
    """Encode with a real codec and decode back to 16 kHz mono, same length."""
    args, container = next((a, c) for n, a, c in CODECS if n == name)
    raw = np.clip(wave, -1, 1).astype("<f4").tobytes()
    encoded = _run(["-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0", *args, "-f", container, "pipe:1"], raw)
    decoded = np.frombuffer(_run(["-i", "pipe:0", "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"], encoded),
                            dtype="<f4")
    out = np.zeros(len(wave), np.float32)
    out[: min(len(wave), len(decoded))] = decoded[: len(wave)]
    return out


def add_noise(wave: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    """White or pink-ish (integrated) noise at the given SNR."""
    noise = rng.standard_normal(len(wave)).astype(np.float32)
    if rng.random() < 0.5:
        noise = np.cumsum(noise)
        noise -= np.convolve(noise, np.ones(64) / 64, mode="same")
    power = np.mean(wave ** 2) + 1e-10
    noise *= np.sqrt(power / (np.mean(noise ** 2) + 1e-10) / 10 ** (snr_db / 10))
    return wave + noise


def edge_silence(wave: np.ndarray, rng: np.random.Generator, max_s=0.6, p=0.5) -> np.ndarray:
    """With probability p, add 0..max_s s of very quiet noise at the start and/or end.

    Applied to genuine and fake recordings alike, after trimming, so the presence
    of a quiet onset or tail carries no label information (ASVspoof 5 train: only
    genuine files have edge silence)."""
    if rng.random() < p:
        # Random raised-cosine fades (0.1-0.8 s) so how a recording ends carries
        # no label information either (genuine speech declines at the end).
        wave = wave.copy()
        for side in (0, 1):
            n = min(int(rng.uniform(0.1, 0.8) * SR), len(wave) // 3)
            if n > 1 and rng.random() < 0.5:
                ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))).astype(np.float32)
                if side == 0:
                    wave[:n] *= ramp
                else:
                    wave[-n:] *= ramp[::-1]
    if rng.random() >= p:
        return wave
    level = 10 ** (rng.uniform(-75, -50) / 20)
    pieces = []
    for side in ("start", "end"):
        n = int(rng.uniform(0, max_s) * SR) if rng.random() < 0.75 else 0
        pieces.append((rng.standard_normal(n) * level).astype(np.float32))
    return np.concatenate([pieces[0], wave, pieces[1]])


def augment(wave: np.ndarray, rng: np.random.Generator, p_codec=0.6, p_noise=0.3, p_gain=0.5):
    """Random chain: gain -> noise -> codec. Returns (wave, description)."""
    steps = []
    if rng.random() < p_gain:
        db = rng.uniform(-8, 4)
        wave = wave * 10 ** (db / 20)
        steps.append(f"gain{db:+.0f}dB")
    if rng.random() < p_noise:
        snr = rng.uniform(10, 40)
        wave = add_noise(wave, rng, snr)
        steps.append(f"noise{snr:.0f}dB")
    wave = np.clip(wave, -1, 1)
    if rng.random() < p_codec:
        name = CODECS[rng.integers(len(CODECS))][0]
        try:
            wave = apply_codec(wave, name)
            steps.append(name)
        except subprocess.CalledProcessError:
            steps.append(f"{name}_failed")
    return wave.astype(np.float32), "+".join(steps) or "clean"
