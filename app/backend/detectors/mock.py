import hashlib
import math
import os

from ..media import probe_duration
from .base import AnalysisResult, ModelInfo, Segment, Verdict

SCORES = {'likely_real': 0.15, 'uncertain': 0.5, 'likely_fake': 0.85}


class MockDetector:
    def __init__(self):
        self.info: ModelInfo = {
            'name': 'mock', 'version': '1.0', 'experimental': True,
            'description': 'Deterministic demonstration model.',
        }
        self.forced_verdict = os.getenv('MOCK_VERDICT')
        if self.forced_verdict not in (None, *SCORES, 'insufficient_audio'):
            raise ValueError('Invalid MOCK_VERDICT')

    def analyze(self, data: bytes, filename: str) -> AnalysisResult:
        duration = probe_duration(data, filename)
        digest = hashlib.sha256(data).digest()
        score = digest[0] / 255
        verdict: Verdict
        if self.forced_verdict:
            verdict = self.forced_verdict
        elif duration < 1:
            verdict = 'insufficient_audio'
        elif score < 0.33:
            verdict = 'likely_real'
        elif score < 0.67:
            verdict = 'uncertain'
        else:
            verdict = 'likely_fake'

        segments: list[Segment] = []
        if verdict == 'insufficient_audio':
            score = None
        else:
            if self.forced_verdict:
                score = SCORES[verdict]
            count = min(600, math.ceil(duration))
            step = duration / count
            for index in range(count):
                offset = digest[index % len(digest)] / 255 - 0.5
                segments.append({
                    'start_s': index * step, 'end_s': min(duration, (index + 1) * step),
                    'fake_probability': max(0, min(1, score + offset * 0.6)),
                })

        return {
            'verdict': verdict, 'fake_probability': score, 'segments': segments,
            'duration_s': duration, 'speech_s': duration if segments else 0,
            'warnings': ['Very short recording'] if duration < 1 else [], 'model': self.info,
        }
