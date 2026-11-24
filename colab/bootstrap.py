"""Run in the owner's mounted Colab session to restore the branch's code bundle."""

import os
import subprocess
import sys
from pathlib import Path

data = Path(os.environ.get("V3_DATA", "/content/drive/MyDrive/audio-v3"))
repo = Path(os.environ.get("V3_REPO", "/content/v3-src"))
repo.mkdir(parents=True, exist_ok=True)
subprocess.run(["tar", "-xf", str(data / "v3-code.tar"), "-C", str(repo)], check=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(repo / "pipeline_v3/requirements-colab.txt"),
    ],
    check=True,
)
