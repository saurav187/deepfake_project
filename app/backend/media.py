import json
import math
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

INPUT_OPTIONS = [
    '-protocol_whitelist', 'file',
    '-format_whitelist', 'wav,mp3,mov,aac,ogg,flac,matroska,webm',
]


@contextmanager
def temporary_audio(data: bytes, filename: str):
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix.lower(), delete=False) as file:
        path = Path(file.name)
        try:
            file.write(data)
            file.close()
            yield path
        finally:
            path.unlink(missing_ok=True)


def probe_duration(data: bytes, filename: str) -> float:
    with temporary_audio(data, filename) as path:
        try:
            probe = subprocess.run(
                ['ffprobe', '-v', 'error', *INPUT_OPTIONS, '-select_streams', 'a:0',
                 '-show_entries', 'stream=duration:format=duration',
                 '-of', 'json', str(path)],
                capture_output=True, check=True, timeout=10,
            )
            metadata = json.loads(probe.stdout)
            stream = metadata['streams'][0]
            duration = stream.get('duration', 'N/A')
            if duration == 'N/A':
                duration = metadata['format']['duration']
            seconds = float(duration)
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError
            return seconds
        except (subprocess.SubprocessError, KeyError, IndexError, ValueError, TypeError):
            raise ValueError('Unreadable audio') from None
