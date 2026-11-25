"""Offline download planner. Prints a resumable shell command for owner review."""

import argparse
import json
import shlex
from pathlib import Path

ZENODO = "https://zenodo.org/api/records"
HF = "https://huggingface.co/datasets"
# Transfer sizes are bytes. Zenodo supplies MD5, Hugging Face LFS supplies SHA-256.
FILES = {
    "librisevoc": (
        f"{ZENODO}/15127251/files/LibriSeVoc.zip/content",
        "LibriSeVoc.zip",
        45046253997,
        "md5",
        "bee5261df62de16ef900861358afae08",
    ),
    "codecfake-sample": (
        f"{HF}/rogertseng/CodecFake/resolve/ae224c9cff0e7a19edf885a670c2c4d1742d3e1e7/data/train-00000-of-00161.parquet",
        "train-00000-of-00161.parquet",
        486611663,
        "sha256",
        "0df27f3b3f361889256d365fe72d4e8fcf9afbeee6cad2901fe6197845d25b84",
    ),
    "indictts-sample": (
        f"{HF}/SherryT997/IndicTTS-Deepfake-Challenge-Data/resolve/57347517658ae989597d8cef303cffb647ed2434/data/train-00000-of-00035.parquet",
        "train-00000-of-00035.parquet",
        453004617,
        "sha256",
        "1b5ddf567b067f2417e0bc1fe620734105b1e68f7f6ce5692c396f07013acb19",
    ),
    "asvspoof19-la": (
        f"{ZENODO}/6906306/files/LA.zip/content",
        "LA.zip",
        7640952520,
        "md5",
        "30c98f11d8b2bc21f2c257bfd78bb5c5",
    ),
}


def plan(root, source):
    url, name, size, algorithm, checksum = FILES[source]
    directory = root.absolute() / "development" / source
    cmd = f"""set -eu
mkdir -p -- {shlex.quote(str(directory))}
cd -- {shlex.quote(str(directory))}
file={shlex.quote(name)}
verify() {{
    test "$(wc -c < "$1")" -eq {size} &&
    printf '%s  %s\\n' {checksum} "$1" | {algorithm}sum --check --status
}}
if test -e "$file"; then
    verify "$file"
else
    part="$file.part"
    if ! test -f "$part" || test "$(wc -c < "$part")" -lt {size}; then
        curl --fail --location --retry 3 --continue-at - --output "$part" {shlex.quote(url)}
    fi
    verify "$part"
    mv -- "$part" "$file"
fi"""
    return {
        "source": source,
        "url": url,
        "bytes": size,
        algorithm: checksum,
        "command": cmd,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", choices=FILES, default=["librisevoc"])
    parser.add_argument(
        "--dry-run", action="store_true", help="explicit offline plan (default)"
    )
    args = parser.parse_args()
    files = [plan(args.data, source) for source in dict.fromkeys(args.sources)]
    print(
        json.dumps(
            {
                "dry_run": True,
                "total_bytes": sum(f["bytes"] for f in files),
                "files": files,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
