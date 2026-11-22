"""Decode every record once and write runs/manifests/<source>.csv.

Usage (repo root):  venv/bin/python -m detector.data.build_manifest [source ...]
With no arguments, every available source is processed. Rows already in a
source's manifest are kept, so re-running after a new ASVspoof 5 tar finishes
only decodes the new files.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import os
import subprocess
import sys
from time import perf_counter

import numpy as np

from detector import paths
from detector.audio import SR, decode, frame_db, pcm_hash, speech_bounds
from detector.data.sources import ADAPTERS, available, read_bytes

FIELDS = ["uid", "source", "locator", "label", "group", "attack", "role",
          "container", "subtype", "rate", "channels", "stored_duration_s",
          "decoded_samples", "speech_s", "lead_silence_s", "trail_silence_s",
          "peak_dbfs", "active_dbfs", "pcm_sha256", "error"]


def describe(record):
    row = record.row()
    try:
        wave, info = decode(read_bytes(record.locator))
        start, end = speech_bounds(wave)
        db = frame_db(wave)
        row.update(container=info.container, subtype=info.subtype, rate=info.rate, channels=info.channels,
                   stored_duration_s=round(info.duration_s, 4), decoded_samples=len(wave),
                   speech_s=round((end - start) / SR, 4), lead_silence_s=round(start / SR, 4),
                   trail_silence_s=round((len(wave) - end) / SR, 4),
                   peak_dbfs=round(float(20 * np.log10(max(np.abs(wave).max(initial=0), 1e-6))), 2),
                   active_dbfs=round(float(np.median(db[db >= db.max() - 40])), 2),
                   pcm_sha256=pcm_hash(wave), error="")
        if len(wave) == 0:
            row["error"] = "empty"
    except (ValueError, RuntimeError, subprocess.CalledProcessError, KeyError, OSError) as exc:
        row.update({k: "" for k in FIELDS if k not in row}, error=f"{type(exc).__name__}: {exc}"[:300])
    return row


def build(source, workers):
    out = paths.MANIFESTS / f"{source}.csv"
    done = set()
    if out.exists():
        with out.open() as f:
            done = {r["uid"] for r in csv.DictReader(f)}
    records = [r for r in ADAPTERS[source]() if r.uid not in done]
    print(f"{source}: {len(done)} already in manifest, {len(records)} to decode", flush=True)
    if not records:
        return
    new_file = not out.exists()
    start = perf_counter()
    with out.open("a", newline="") as f, ThreadPoolExecutor(workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, row in enumerate(pool.map(describe, records, buffersize=4 * workers), 1):
            writer.writerow(row)
            if i % 2000 == 0 or i == len(records):
                rate = i / (perf_counter() - start)
                print(f"  {source}: {i}/{len(records)} ({rate:.0f} files/s)", flush=True)
                f.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", nargs="*", help=f"any of {list(ADAPTERS)}")
    p.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = p.parse_args()
    paths.MANIFESTS.mkdir(parents=True, exist_ok=True)
    ready = available()
    for source in args.sources or ready:
        if source not in ready:
            print(f"{source}: data not downloaded/verified yet, skipped", file=sys.stderr)
            continue
        build(source, args.workers)


if __name__ == "__main__":
    main()
