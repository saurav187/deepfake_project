"""One canonical audio path for training, evaluation and the app.

Every input, whatever its container, rate or channel count, becomes 16 kHz mono
float32 through the same code. Lesson from the legacy pipeline: the source
format predicted the label, so format must never reach the model, and the
same decode must run everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import soxr

SR = 16000
EDGE_CUT_S = 0.25   # see prepare()
EDGE_FADE_S = 0.15


@dataclass
class SourceInfo:
    """Properties of the stored file, kept for the leakage audit only."""
    container: str
    subtype: str
    rate: int
    channels: int
    duration_s: float


def _ffmpeg_decode(path: str) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
         "-ar", str(SR), "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<f4").copy()


def _ffprobe(path: str) -> SourceInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels,duration:format=format_name,duration",
         "-of", "default=noprint_wrappers=1", path],
        capture_output=True, check=True, text=False).stdout.decode()
    kv = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    dur = kv.get("duration", "N/A")
    return SourceInfo(container=kv.get("format_name", "?"), subtype=kv.get("codec_name", "?"),
                      rate=int(kv.get("sample_rate", 0) or 0), channels=int(kv.get("channels", 0) or 0),
                      duration_s=float(dur) if dur not in ("N/A", "") else float("nan"))


def decode(data: bytes) -> tuple[np.ndarray, SourceInfo]:
    """Decode any audio bytes to canonical 16 kHz mono float32 in [-1, 1]."""
    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            info = SourceInfo(container=f.format.lower(), subtype=f.subtype.lower(), rate=f.samplerate,
                              channels=f.channels, duration_s=f.frames / f.samplerate)
            audio = f.read(dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if info.rate != SR:
            mono = soxr.resample(mono, info.rate, SR, quality="HQ")
        wave = np.ascontiguousarray(mono, dtype=np.float32)
    except (sf.LibsndfileError, RuntimeError):
        # MP3/AAC/Opus/M4A and anything libsndfile rejects. MP4-family files need
        # a seekable input, so ffmpeg reads a temporary file instead of a pipe.
        with tempfile.NamedTemporaryFile(suffix=".audio") as tmp:
            tmp.write(data)
            tmp.flush()
            info = _ffprobe(tmp.name)
            wave = _ffmpeg_decode(tmp.name)
        if len(wave) == 0:
            raise ValueError("ffmpeg decoded no audio samples")
    if not np.isfinite(wave).all():
        raise ValueError("decoded audio contains non-finite samples")
    return np.clip(wave, -1.0, 1.0), info


def pcm_hash(wave: np.ndarray) -> str:
    """Content identity of decoded audio (exact duplicates across files/sources)."""
    q = np.round(np.clip(wave, -1, 1) * 32767).astype("<i2")
    return hashlib.sha256(q.tobytes()).hexdigest()


def frame_db(wave: np.ndarray, frame: int = 320) -> np.ndarray:
    """RMS level in dBFS of non-overlapping 20 ms frames."""
    n = len(wave) // frame
    if n == 0:
        return np.full(1, -120.0)
    rms = np.sqrt(np.mean(wave[: n * frame].reshape(n, frame) ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-6))


def speech_bounds(wave: np.ndarray, rel_db: float = 40.0, floor_db: float = -60.0,
                  margin_s: float = 0.1) -> tuple[int, int]:
    """Sample range from the first to the last frame within rel_db of the loudest frame.

    Silence length is a known label shortcut (e.g. in ASVspoof 2019), so leading
    and trailing silence is cut before the model ever sees the audio.
    """
    db = frame_db(wave)
    active = np.flatnonzero(db >= max(db.max() - rel_db, floor_db))
    if len(active) == 0:
        return 0, 0
    margin = int(margin_s * SR)
    return max(0, active[0] * 320 - margin), min(len(wave), (active[-1] + 1) * 320 + margin)


def prepare(wave: np.ndarray, target_dbfs: float = -25.0) -> np.ndarray:
    """Trim edge silence and normalise loudness of the active frames.

    The trim has no margin. In ASVspoof 5 train, genuine files carry leading and
    trailing silence (median 0.16 s / 0.22 s) while fakes have none, so any kept
    margin would leave a quiet onset that marks genuine audio."""
    return prepare_with_offset(wave, target_dbfs)[0]


def prepare_with_offset(wave: np.ndarray, target_dbfs: float = -25.0) -> tuple[np.ndarray, int]:
    """prepare(), also returning where the prepared audio starts in the input (samples),
    so timeline segments can be reported at their original positions."""
    start, end = speech_bounds(wave, margin_s=0.0)
    wave = wave[start:end]
    if len(wave) == 0:
        return wave, start
    # Fakes in ASVspoof 5 train stop abruptly at full level, while genuine speech
    # decays naturally (tail -3.1 vs -7.0 dB re. active level). Dropping 0.25 s at
    # each edge and fading the new edges over 0.15 s brought every edge-level and
    # edge-slope cue to AUC <= 0.60 on a 2,400-recording train sample; dropping
    # more did not help. Short clips lose proportionally less.
    cut = min(EDGE_CUT_S, 0.1 * len(wave) / SR)
    offset = start + int(cut * SR)
    wave = wave[int(cut * SR): len(wave) - int(cut * SR)]
    db = frame_db(wave)
    active = db >= db.max() - 40
    level = 10 * np.log10(np.mean(10 ** (db[active] / 10))) if active.any() else -120.0
    gain = 10 ** ((target_dbfs - level) / 20)
    wave = wave * gain
    peak = np.abs(wave).max()
    wave = (wave / peak * 0.99 if peak > 0.99 else wave).astype(np.float32)
    n = min(int(EDGE_FADE_S * SR), len(wave) // 4)
    if n > 1:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))).astype(np.float32)
        wave[:n] *= ramp
        wave[-n:] *= ramp[::-1]
    return wave, offset


def window_starts(n_samples: int, size: int, max_windows: int | None = None) -> list[int]:
    """The one window schedule shared by dev scoring, calibration, evaluation and the app.

    Evenly spaced windows of `size` samples whose union covers the whole recording
    (the last ends at the end); `max_windows` caps the count (dev uses 4, which
    covers recordings up to 16 s completely). Callers repeat-pad recordings shorter
    than one window to `size` first, so n_samples >= size."""
    n = int(np.ceil(n_samples / size))
    if max_windows:
        n = min(n, max_windows)
    return np.linspace(0, n_samples - size, n).astype(int).tolist() if n > 1 else [0]


def segments(wave: np.ndarray, seconds: float = 4.0, hop: float | None = None):
    """Yield (start_s, end_s, window). Short tails repeat the audio instead of
    zero padding, so silence padding cannot become evidence."""
    size = int(seconds * SR)
    step = int((hop or seconds) * SR)
    if len(wave) == 0:
        return
    if len(wave) <= size:
        reps = int(np.ceil(size / len(wave)))
        yield 0.0, len(wave) / SR, np.tile(wave, reps)[:size]
        return
    for start in range(0, len(wave) - size + 1, step):
        yield start / SR, (start + size) / SR, wave[start:start + size]
    tail = len(wave) - size
    if tail % step:
        yield tail / SR, len(wave) / SR, wave[tail:]
