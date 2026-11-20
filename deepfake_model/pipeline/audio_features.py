"""Canonical 16 kHz mono input and one-second log-Mel windows."""
from pathlib import Path
import subprocess

import librosa
import numpy as np
import soundfile as sf

SR = 16000
BASE = Path(__file__).resolve().parents[1]
RUN = BASE / 'pipeline' / 'outputs' / 'audited_v1'
ROOT = BASE / 'datasets' / 'indiefake'


def read_audio(path):
    try:
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
    except sf.LibsndfileError:
        # External recordings include AAC in M4A containers.
        data = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path),
                                        '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'])
        return np.frombuffer(data, dtype='<f4')
    mono = audio.mean(axis=1)
    return librosa.resample(mono, orig_sr=rate, target_sr=SR) if rate != SR else mono


def mel_power(audio):
    return librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=400, hop_length=160,
                                         win_length=400, n_mels=128, fmax=8000)


def feature(audio):
    if len(audio) < SR:
        audio = np.pad(audio, (0, SR-len(audio)))
    return np.ascontiguousarray(np.clip(librosa.power_to_db(mel_power(audio), ref=np.max),
                                        -80, 0)[:, :94], dtype=np.float32)


def windows(audio):
    for start in range(0, len(audio), SR):
        yield audio[start:start+SR]
