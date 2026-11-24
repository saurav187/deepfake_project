"""Seekable, anonymous memory for FFmpeg on the Linux app/Colab target."""

import json
import math
import os
import subprocess
from contextlib import contextmanager

FORMATS = "wav,mp3,mov,aac,ogg,flac,matroska,webm"


@contextmanager
def input_fd(data):
    with os.fdopen(os.memfd_create("audio"), "w+b") as f:
        f.write(data)
        f.seek(0)
        yield f.fileno()


def read_media(cmd, data, options):
    with input_fd(data) as fd:
        try:
            return subprocess.run(
                [
                    cmd,
                    "-v",
                    "error",
                    "-protocol_whitelist",
                    "file,pipe",
                    "-format_whitelist",
                    FORMATS,
                    "-i",
                    f"/proc/self/fd/{fd}",
                    *options,
                ],
                pass_fds=(fd,),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=60,
                check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("Unreadable audio") from exc


def probe_duration(data, filename):
    metadata = json.loads(
        read_media(
            "ffprobe",
            data,
            [
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=duration:format=duration",
                "-of",
                "json",
            ],
        )
    )
    try:
        duration = metadata["streams"][0].get("duration", "N/A")
        if duration == "N/A":
            duration = metadata["format"]["duration"]
        seconds = float(duration)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError
        return seconds
    except (KeyError, IndexError, TypeError, ValueError):
        raise ValueError("Unreadable audio") from None
