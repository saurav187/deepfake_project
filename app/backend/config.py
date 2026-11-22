import os
from dataclasses import dataclass

EXTENSIONS = ('.wav', '.mp3', '.m4a', '.aac', '.ogg', '.opus', '.flac', '.webm', '.mp4', '.mov')


@dataclass(frozen=True)
class Settings:
    backend: str = 'mock'
    max_upload_mb: int = 50
    max_duration_s: int = 600
    max_queue: int = 10
    workers: int | None = None
    result_ttl_s: float = 3600
    allowed_origin: str | None = None
    device: str = 'auto'

    @classmethod
    def from_env(cls):
        workers = os.getenv('ANALYSIS_WORKERS')
        return cls(
            backend=os.getenv('DETECTOR_BACKEND', 'mock'),
            max_upload_mb=int(os.getenv('MAX_UPLOAD_MB', '50')),
            max_duration_s=int(os.getenv('MAX_DURATION_S', '600')),
            max_queue=int(os.getenv('MAX_QUEUE', '10')),
            workers=int(workers) if workers else None,
            result_ttl_s=float(os.getenv('RESULT_TTL_S', '3600')),
            allowed_origin=os.getenv('ALLOWED_ORIGIN'),
            device=os.getenv('DEVICE', 'auto'),
        )

