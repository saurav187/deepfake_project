import importlib
import math
import subprocess
import sys
from pathlib import Path
from statistics import fmean

from ..media import INPUT_OPTIONS, temporary_audio
from .base import AnalysisResult, ModelInfo, Segment, Verdict
from .registry import BackendUnavailable

PIPELINE = Path(__file__).resolve().parents[3] / 'deepfake_model' / 'pipeline'
CHECKPOINT = PIPELINE / 'outputs' / 'audited_v1' / 'full' / 'checkpoint.pth'


class LegacyDetector:
    def __init__(self, device: str):
        if not CHECKPOINT.is_file() or not CHECKPOINT.with_name('rule.json').is_file():
            raise BackendUnavailable('Legacy detector requires the audited checkpoint and rule.json.')
        previous = sys.dont_write_bytecode
        sys.path.insert(0, str(PIPELINE))
        try:
            sys.dont_write_bytecode = True
            self.pipeline = importlib.import_module('04_realtime_inference')
        finally:
            sys.dont_write_bytecode = previous
            sys.path.remove(str(PIPELINE))
        self.pipeline.torch.set_num_threads(2)
        self.detector = self.pipeline.Detector(CHECKPOINT, device=None if device == 'auto' else device)
        self.device = self.detector.device
        self.threshold = float(self.detector.rule['threshold'])
        self.info: ModelInfo = {
            'name': 'legacy-cnn-lstm',
            'version': str(self.detector.rule['version']),
            'experimental': True,
            'description': 'Research prototype; unreliable on real-world audio',
        }

    def decode(self, path: Path):
        import librosa
        import numpy as np
        import soundfile as sf

        # Match the frozen read_audio while keeping ffmpeg diagnostics private.
        target_rate = self.pipeline.SR
        try:
            audio, rate = sf.read(path, dtype='float32', always_2d=True)
        except sf.LibsndfileError:
            data = subprocess.check_output(
                ['ffmpeg', *INPUT_OPTIONS, '-v', 'error', '-i', str(path),
                 '-f', 'f32le', '-ac', '1', '-ar', str(target_rate), '-'],
                stderr=subprocess.DEVNULL, timeout=60,
            )
            return np.frombuffer(data, dtype='<f4')
        mono = audio.mean(axis=1)
        return librosa.resample(mono, orig_sr=rate, target_sr=target_rate) if rate != target_rate else mono

    def analyze(self, data: bytes, filename: str) -> AnalysisResult:
        with temporary_audio(data, filename) as path:
            try:
                audio = self.decode(path)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
                raise ValueError('Unreadable audio') from None
        if not len(audio):
            raise ValueError('Unreadable audio')

        stream = self.pipeline.Stream(self.detector)
        rate = self.pipeline.SR
        for start in range(0, len(audio), 32 * rate):
            stream.feed(audio[start:start + 32 * rate])
        duration = float(stream.finish()['duration_s'])
        segments: list[Segment] = [
            {'start_s': float(math.ceil(end) - 1), 'end_s': float(end), 'fake_probability': float(score)}
            for end, score in zip(stream.ends, stream.scores, strict=True)
        ]
        probability = fmean(stream.scores) if segments else None
        verdict: Verdict
        if probability is None:
            verdict = 'insufficient_audio'
        elif probability >= self.threshold:
            verdict = 'likely_fake'
        elif probability < self.threshold - 0.2:
            verdict = 'likely_real'
        else:
            verdict = 'uncertain'
        return {
            'verdict': verdict, 'fake_probability': probability, 'segments': segments,
            'duration_s': duration, 'speech_s': sum((s['end_s'] - s['start_s'] for s in segments), 0.0),
            'warnings': [], 'model': self.info,
        }
