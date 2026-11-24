"""Execute one notebook stage from the CLI. Mount Drive and unpack source first."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(stage, repo, data, local, execute=False):
    commands = {
        "setup": ["plan"],
        "prepare": ["inventory", "match"],
        "pack": ["catalog", "pack"],
        "pilot": ["stage", "audit", "pilot"],
        "teacher": ["stage", "teacher"],
        "student": ["stage", "student", "export"],
        "evaluate": ["stage", "evaluate", "benchmark"],
    }
    stages = {
        "pilot": ["train", "select"],
        "teacher": ["train", "select"],
        "student": ["train", "select", "calibrate", "threshold"],
        "evaluate": ["validate"],
    }
    for action in commands[stage]:
        cmd = [
            sys.executable,
            "-m",
            "pipeline_v3.cli",
            action,
            "--data",
            str(data),
            "--local",
            str(local),
        ]
        if action == "stage":
            cmd.extend(["--splits", *stages[stage]])
        if not execute:
            cmd.append("--dry-run")
        subprocess.run(cmd, cwd=repo, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=["setup", "prepare", "pack", "pilot", "teacher", "student", "evaluate"],
        default=os.environ.get("V3_STAGE", "setup"),
    )
    p.add_argument(
        "--repo", type=Path, default=Path(os.getenv("V3_REPO", "/content/v3-src"))
    )
    p.add_argument(
        "--data",
        type=Path,
        default=Path(os.getenv("V3_DATA", "/content/drive/MyDrive/audio-v3")),
    )
    p.add_argument(
        "--local", type=Path, default=Path(os.getenv("V3_LOCAL", "/content/v3-local"))
    )
    p.add_argument(
        "--execute", action="store_true", default=os.getenv("V3_EXECUTE") == "1"
    )
    # colab exec executes code in an IPython kernel. Its argv belongs to ipykernel.
    args = p.parse_args(
        []
        if "IPython" in sys.modules and Path(sys.argv[0]).name != "run_stage.py"
        else None
    )
    run(args.stage, args.repo, args.data, args.local, args.execute)
