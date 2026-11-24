#!/usr/bin/env python3
"""Read-only progress monitor for DOWNLOAD_PLAN.sh."""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from shutil import disk_usage
from time import monotonic, sleep
import sys

GB = 10**9
# Rounded transfer sizes from DOWNLOAD_PLAN.sh --sizes.
SELECTIONS = {
    "development": {
        "mailabs": (1, 8.0168),
        "mlaad": (4004, 1.3593),
        "kathbath": (32, 15.9758),
        "indicsynth": (14, 7.1216),
        "indicvoices": (5, 2.9351),
        "shrutilipi": (2, 0.9898),
        "speechfake": (7, 136.3721),
    },
    "sealed": {
        "deepfake-eval": (1981, 4.0657),
        "voxenes": (1, 19.8468),
        "llamapartial": (7, 28.2200),
    },
}


def scan(directory):
    """Count payloads and bytes written to resumable partials, excluding metadata."""
    files = complete = partial = 0
    for path in directory.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            info = path.stat()
        except FileNotFoundError:
            continue  # Downloader renamed the partial between listing and stat.
        if path.name.endswith((".part", ".incomplete")):
            partial += min(info.st_size, info.st_blocks * 512)
        elif ".cache" not in path.relative_to(directory).parts:
            files += 1
            complete += info.st_size
    return files, complete, partial


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Directory passed to DOWNLOAD_PLAN.sh")
    parser.add_argument("--interval", type=float, default=5, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    args = parser.parse_args()
    if not args.data.is_dir():
        parser.error("DATA is missing. Check that the hard disk is mounted.")
    if not 0 < args.interval < float("inf"):
        parser.error("--interval must be a positive, finite number")

    previous = None
    while True:
        now = monotonic()
        total = 0
        lines = [
            f"Downloads at {datetime.now():%H:%M:%S}  |  Disk free {disk_usage(args.data).free / GB:.1f} GB",
            f"{'Dataset':<16} {'Files on disk':>13} {'Ready GB':>10} {'Partial GB':>11} {'Plan GB':>9} {'On disk':>8}",
        ]
        for folder, datasets in SELECTIONS.items():
            for key, (expected_files, expected_gb) in datasets.items():
                files, complete, partial = scan(args.data / folder / key)
                total += complete + partial
                percent = min(100, 100 * (complete + partial) / (expected_gb * GB))
                lines.append(
                    f"{key:<16} {f'{files}/{expected_files}':>13} {complete / GB:>10.3f}"
                    f" {partial / GB:>11.3f} {expected_gb:>9.3f} {percent:>7.1f}%"
                )
        speed = ""
        if previous is not None:
            last_time, last_total = previous
            mb_per_second = (total - last_total) / (now - last_time) / 10**6
            speed = f"  |  Recent disk growth {max(0, mb_per_second):.2f} MB/s"
        expected = sum(size for group in SELECTIONS.values() for _, size in group.values())
        lines.append(f"Total on disk {total / GB:.2f} / {expected:.2f} GB{speed}")
        if not args.once and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print("\n".join(lines), flush=True)
        if args.once:
            return
        previous = now, total
        sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
