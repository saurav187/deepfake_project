"""Score ASVspoof 2021 LA eval/notrim none and ulaw trials directly from tar.gz."""
import argparse
import csv
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch

from audio_features import BASE, RUN, SR
from prepare_manifest import digest, write_csv

inference = importlib.import_module("04_realtime_inference")
DATA = BASE / "datasets/asvspoof2021_la"
AUDIO = DATA / "ASVspoof2021_LA_eval.tar.gz"
MD5 = "2abee34d8b0b91159555fc4f016e4562"
KEYS = DATA / "LA-keys-full.tar.gz"
PILOT_LIMIT = 2048


def protocol():
    columns = "speaker utterance codec transmission attack label trim phase".split()
    trials = {}
    with tarfile.open(KEYS) as archive:
        for line in archive.extractfile("keys/LA/CM/trial_metadata.txt"):
            row = dict(zip(columns, line.decode().split(), strict=True))
            if row["phase"] == "eval" and row["trim"] == "notrim" and row["codec"] in {"ulaw", "none"}:
                uid = row["utterance"]
                if uid in trials or row["label"] not in {"bonafide", "spoof"}:
                    raise ValueError(row)
                trials[uid] = row
    return trials


def decode_flac(raw, uid):
    # STREAMINFO carries the MD5 of little-endian PCM. Some official FLAC
    # streams fail in the system libFLAC but decode sample-exactly in ffmpeg.
    if raw[:4] != b"fLaC" or raw[4] & 127 or int.from_bytes(raw[5:8]) != 34:
        raise ValueError(f"Missing FLAC STREAMINFO: {uid}")
    info = int.from_bytes(raw[18:26])
    rate = info >> 44
    channels = 1 + ((info >> 41) & 7)
    bits = 1 + ((info >> 36) & 31)
    frames = info & ((1 << 36) - 1)
    if (rate, channels, bits) != (SR, 1, 16) or not frames:
        raise ValueError(f"Unexpected stream format: {uid}")
    expected_md5 = raw[26:42]
    try:
        samples, _ = sf.read(io.BytesIO(raw), dtype="int16")
        pcm = samples.astype("<i2", copy=False).tobytes()
    except sf.LibsndfileError:
        pcm = b""
    if hashlib.md5(pcm).digest() != expected_md5:
        pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", "pipe:0",
                              "-f", "s16le", "pipe:1"], input=raw, capture_output=True,
                             check=True).stdout
    if len(pcm) != 2 * frames or hashlib.md5(pcm).digest() != expected_md5:
        raise ValueError(f"Decoded PCM checksum or length mismatch: {uid}")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768


def score(archive_path, pilot):
    with (RUN / "manifest.csv").open() as f:
        known = {r["pcm_hash"] for r in csv.DictReader(f)}
    trials = protocol()
    detector = inference.Detector(device="cpu")
    results = []
    seen = set()
    files = 0
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for entry in archive:
            if not entry.isfile() or not entry.name.endswith(".flac"):
                continue
            uid = Path(entry.name).stem
            if uid in seen:
                raise ValueError(f"Duplicate audio ID {uid}")
            seen.add(uid)
            files += 1
            if uid in trials:
                raw = archive.extractfile(entry).read()
                try:
                    audio = decode_flac(raw, uid)
                except (ValueError, subprocess.CalledProcessError) as exc:
                    results.append(dict(**trials[uid], matches_baseline=None, error=str(exc),
                                        duration_s=None, score=None, verdict="DECODE_ERROR",
                                        alerts=None, first_alert_s=None))
                else:
                    stream = inference.Stream(detector)
                    stream.feed(audio)
                    results.append(dict(**trials[uid], matches_baseline=digest(audio) in known, error="",
                                        **stream.finish()))
            if pilot and files == PILOT_LIMIT:
                break
    if (pilot and files != PILOT_LIMIT) or (not pilot and trials.keys() - seen):
        raise ValueError("Archive did not cover requested trials")
    return files, results


def summarize(rows):
    summary = []
    for codec in ("none", "ulaw"):
        subset = [r for r in rows if r["codec"] == codec and not r["matches_baseline"]]
        errors = [r for r in subset if r["verdict"] == "DECODE_ERROR"]
        subset = [r for r in subset if r["verdict"] != "DECODE_ERROR"]
        genuine = [r for r in subset if r["label"] == "bonafide"]
        spoof = [r for r in subset if r["label"] == "spoof"]
        fp = sum(r["verdict"] == "FAKE" for r in genuine)
        tp = sum(r["verdict"] == "FAKE" for r in spoof)
        summary.append(dict(codec=codec, genuine_decode_errors=sum(r["label"] == "bonafide" for r in errors),
                            spoof_decode_errors=sum(r["label"] == "spoof" for r in errors), genuine=len(genuine), false_positives=fp,
                            false_positive_rate=fp/len(genuine) if genuine else None,
                            spoof=len(spoof), detected=tp, recall=tp/len(spoof) if spoof else None,
                            insufficient=sum(r["verdict"] == "INSUFFICIENT" for r in subset)))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", action="store_true", help="Use the first 2048 archived FLAC members")
    p.add_argument("--archive", type=Path, default=AUDIO)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        p.error("--out must be a new directory")
    if not args.pilot:
        with args.archive.open("rb") as f:
            if hashlib.file_digest(f, "md5").hexdigest() != MD5:
                raise ValueError("Archive differs from the published Zenodo MD5")
    torch.set_num_threads(2)
    files, rows = score(args.archive, args.pilot)
    stats = summarize(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "results"
        stage.mkdir()
        write_csv(stage / "predictions.csv", rows)
        write_csv(stage / "summary.csv", stats)
        rule = json.loads((RUN / "full/rule.json").read_text())
        run = dict(pilot=args.pilot, archived_flacs_visited=files, selected=len(rows),
                   exact_pcm_matches=sum(r["matches_baseline"] is True for r in rows),
                   archive_md5_verified=not args.pilot, rule=rule,
                   pcm_format="mono 16kHz float32", source="https://doi.org/10.5281/zenodo.4837263",
                   keys_url="https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz",
                   keys_sha256=hashlib.sha256(KEYS.read_bytes()).hexdigest())
        (stage / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        stage.rename(args.out)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
