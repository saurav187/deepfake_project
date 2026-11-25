"""In-memory shared decode, preparation and feature extraction."""

import subprocess

import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly

from .media import read_media

SR = 16000
WINDOW = 4 * SR


def ffmpeg(data, args, output):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-protocol_whitelist",
        "pipe",
        *args,
        "-i",
        "pipe:0",
        "-map",
        "0:a:0",
        *output,
        "pipe:1",
    ]
    try:
        return subprocess.run(
            cmd,
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=60,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        raise ValueError("Unreadable audio") from exc


def decode(data):
    raw = read_media(
        "ffmpeg",
        data,
        ["-map", "0:a:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
    )
    x = np.frombuffer(raw, "<f4")
    if not x.size or not np.isfinite(x).all():
        raise ValueError("Unreadable audio")
    return x


def channel(x, name, rng=None):
    if name == "clean":
        return x
    if name in ("opus", "mp3", "aac"):
        codec, fmt = {
            "opus": ("libopus", "ogg"),
            "mp3": ("libmp3lame", "mp3"),
            "aac": ("aac", "adts"),
        }[name]
        data = ffmpeg(
            x.astype("<f4").tobytes(),
            ["-f", "f32le", "-ar", str(SR), "-ac", "1"],
            ["-c:a", codec, "-b:a", "16k" if name == "opus" else "32k", "-f", fmt],
        )
        return decode(data)[: len(x)]
    if name == "phone":
        return resample_poly(resample_poly(x, 1, 2), 2, 1)[: len(x)]
    if name == "rate":
        return resample_poly(resample_poly(x, 3, 2), 2, 3)[: len(x)]
    if name == "playback":
        y = x.copy()
        y[400:] += 0.22 * x[:-400]
        return y
    if name == "rawboost":
        # Impulsive signal-dependent noise (RawBoost algorithm 2).
        y = x.copy()
        count = int(rng.uniform(0, 0.05) * len(x))
        pos = rng.choice(len(x), count, replace=False)
        y[pos] *= 1 + 2 * rng.uniform(-1, 1, count) * rng.uniform(-1, 1, count)
        return y / max(float(np.max(np.abs(y))), 1.0)
    if name == "noise":
        return (x + rng.normal(0, max(np.std(x), 1e-6) * 0.03, len(x))).astype(
            "float32"
        )
    raise ValueError(f"Unknown channel {name}")


def prepare(x):
    # Energy gate estimates the speech span. Interior pauses retain their timestamps.
    hop = SR // 100
    frames = np.pad(x, (0, (-len(x)) % hop)).reshape(-1, hop)
    rms = np.sqrt((frames**2).mean(axis=1))
    active = np.flatnonzero(rms > max(1e-4, np.max(rms, initial=0) * 0.025))
    if not active.size:
        return x[:0], 0
    start, stop = int(active[0]) * hop, min(len(x), (int(active[-1]) + 1) * hop)
    cut = min(SR // 4, (stop - start) // 12)
    start, stop = start + cut, stop - cut
    y = x[start:stop].copy()
    fade = min(int(0.15 * SR), len(y) // 4)
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    return y / max(float(y.std()), 1e-5), start


def windows(x):
    if len(x) < SR:
        return
    last = max(len(x) - WINDOW, 0)
    for start in sorted({*range(0, last + 1, WINDOW), last}):
        yield start, x[start : start + WINDOW]


def features(x):
    """64 log linear-frequency bands on valid frames, no padding in temporal pooling."""
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160]
    power = abs(np.fft.rfft(frames * np.hanning(400), n=512))[:, 1:] ** 2
    return np.log(power.reshape(-1, 64, 4).mean(axis=2) + 1e-7).astype("float32")


def cepstra(x):
    c = dct(features(x), norm="ortho", axis=1)[:, :24]
    return np.concatenate([c.mean(0), c.std(0)])


def fixed_crop(data):
    """Trim both labels identically and return four seconds of unpadded source PCM."""
    import io
    import wave

    x = decode(data)
    trimmed, start = prepare(x)
    if len(trimmed) < WINDOW:
        return None
    offset = start + (len(trimmed) - WINDOW) // 2
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setparams((1, 2, SR, 0, "NONE", ""))
        f.writeframes(
            (np.clip(x[offset : offset + WINDOW], -1, 32767 / 32768) * 32768)
            .round()
            .astype("<i2")
            .tobytes()
        )
    return buf.getvalue()


def model_view(x, row):
    if row["pair"] == "phonemedf" or row.get("prepared_view") == "v2a-fixed-norm":
        if len(x) < WINDOW:
            return x[:0]
        y = x[:WINDOW]
        return y / max(float(y.std()), 1e-5)
    return prepare(x)[0]
