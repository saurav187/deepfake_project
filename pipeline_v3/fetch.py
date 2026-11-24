"""Owner-run downloads to Drive. List and total exact files before transferring any."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MINIMUM = {
    "mailabs": 8.017,
    "mlaad": 1.359,
    "kathbath": 15.976,
    "indicsynth": 7.122,
    "indicvoices": 2.935,
}
EXTRAS = {
    "shrutilipi": 0.990,
    "voxenes": 19.847,
    "deepfake-eval": 4.066,
    "llamapartial": 28.220,
}
ENCODERS = {
    "xlsr": "facebook/wav2vec2-xls-r-300m",
    "wavlm-base": "microsoft/wavlm-base-plus",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument(
        "--sources",
        nargs="+",
        choices=[*MINIMUM, *EXTRAS, *ENCODERS],
        default=list(MINIMUM),
    )
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = p.parse_args()
    selected = list(dict.fromkeys(args.sources))
    sizes = MINIMUM | EXTRAS
    print(
        json.dumps(
            {
                "files": selected,
                "archived_audio_GB": sum(sizes.get(s, 0) for s in selected),
                "model_budget_extra_GB": sum(
                    2 if s in ENCODERS else 0 for s in selected
                ),
                "destination": str(args.data),
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return
    args.data.mkdir(parents=True, exist_ok=True)
    audio = [s for s in selected if s in sizes]
    if audio:
        script = Path(__file__).resolve().parents[1] / "college" / "DOWNLOAD_PLAN.sh"
        subprocess.run(
            ["bash", str(script), str(args.data.resolve()), *audio],
            env=dict(os.environ, PYTHON=sys.executable),
            input="y\n" * len(audio),
            text=True,
            check=True,
        )
    for s in selected:
        if s in ENCODERS:
            from huggingface_hub import HfApi, hf_hub_download

            dest = args.data / "models" / s
            lock = dest / "revision.txt"
            rev = (
                lock.read_text().strip()
                if lock.exists()
                else HfApi().model_info(ENCODERS[s]).sha
            )
            names = HfApi().list_repo_files(ENCODERS[s], revision=rev)
            weights = (
                "model.safetensors"
                if "model.safetensors" in names
                else "pytorch_model.bin"
            )
            dest.mkdir(parents=True, exist_ok=True)
            lock.write_text(rev)
            for filename in ["config.json", "preprocessor_config.json", weights]:
                hf_hub_download(ENCODERS[s], filename, revision=rev, local_dir=dest)


if __name__ == "__main__":
    main()
