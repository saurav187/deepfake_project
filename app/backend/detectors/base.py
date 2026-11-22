from typing import Literal, Protocol, TypedDict

Verdict = Literal['likely_real', 'uncertain', 'likely_fake', 'insufficient_audio']


class Segment(TypedDict):
    start_s: float
    end_s: float
    fake_probability: float


class ModelInfo(TypedDict):
    name: str
    version: str
    experimental: bool
    description: str


class AnalysisResult(TypedDict):
    verdict: Verdict
    fake_probability: float | None
    segments: list[Segment]
    duration_s: float
    speech_s: float
    warnings: list[str]
    model: ModelInfo


class Detector(Protocol):
    info: ModelInfo

    def analyze(self, data: bytes, filename: str) -> AnalysisResult: ...
