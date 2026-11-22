"""Development probe of the frozen detector on source-matched VA excerpts."""
import argparse
import csv
import hashlib
import importlib
import json
from collections import defaultdict
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import torch

from audio_features import BASE, RUN, SR, read_audio
from prepare_manifest import digest, write_csv

inference = importlib.import_module("04_realtime_inference")
VA = BASE / "datasets" / "va_dataset"
SECONDS = 10
BANDLIMIT = "float_pcm_8khz"
CODEC = "mulaw_8khz"


def checksum(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def phone_audio(audio, channel):
    """Apply identical resampling with either linear float PCM or G.711 companding."""
    def convert(data, input_format, input_rate, output_format, output_rate):
        return subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-f", input_format, "-ar", str(input_rate),
             "-ac", "1", "-i", "pipe:0", "-f", output_format, "-ar", str(output_rate),
             "-ac", "1", "pipe:1"], input=data, capture_output=True, check=True,
        ).stdout

    fmt, width = {BANDLIMIT: ("f32le", 4), CODEC: ("mulaw", 1)}[channel]
    encoded = convert(audio.astype("<f4", copy=False).tobytes(), "f32le", SR, fmt, 8000)
    decoded = np.frombuffer(convert(encoded, fmt, 8000, "f32le", SR), dtype="<f4")
    if len(encoded) != width * (len(audio) // 2) or len(decoded) != len(audio) or not np.isfinite(decoded).all():
        raise ValueError("Unexpected channel round trip length or non-finite samples")
    return decoded


def audit(rule):
    manifest = RUN / "manifest.csv"
    if checksum(manifest) != rule["manifest_sha256"]:
        raise ValueError("Frozen split manifest checksum mismatch")
    matches = defaultdict(set)
    with manifest.open() as f:
        for row in csv.DictReader(f):
            matches[row["pcm_hash"]].add((row["source_file"], row["split"]))

    with (RUN / "full/external_predictions.csv").open() as f:
        baseline = {r["source_file"]: r for r in csv.DictReader(f)}
    paths = sorted(p for p in VA.rglob("*") if p.is_file())
    if {p.relative_to(VA).as_posix() for p in paths} != baseline.keys():
        raise ValueError("VA collection differs from the saved development evaluation")
    rows = []
    for path in paths:
        name = path.relative_to(VA).as_posix()
        audio = read_audio(path)
        pcm = digest(audio)
        matched = sorted(matches.get(pcm, ()))
        row = baseline[name]
        if abs(len(audio) / SR - float(row["duration_s"])) > 1 / SR:
            raise ValueError(f"Duration changed for {name}")
        rows.append(dict(file=name, label=int(row["label"]), samples=len(audio), pcm_sha256=pcm,
                         baseline_verdict=row["verdict"], baseline_score=row["score"],
                         matching_splits="|".join(sorted({s for _, s in matched})),
                         matching_sources="|".join(sorted({n for n, _ in matched}))))
    return rows


def matched_trials(detector, audit_rows):
    lookup = {r["file"]: r for r in audit_rows}
    families = defaultdict(list)
    for name in lookup:
        match = re.fullmatch(r"fake/([a-z]+)-to-([a-z]+)\.wav", name.lower())
        if match:
            families[match.group(1)].append(name)
    if not families:
        raise ValueError("No source-named conversions")
    trials = []
    for family, fakes in sorted(families.items()):
        sources = [f"real/{family}-original.wav", *sorted(fakes)]
        if any(lookup[name]["matching_sources"] for name in sources):
            raise ValueError(f"Matched family {family} contains baseline audio")
        shared_seconds = min(lookup[name]["samples"] for name in sources) // SR
        if shared_seconds < 3 * SECONDS:
            raise ValueError(f"Insufficient duration for three disjoint excerpts in {family}")
        last = shared_seconds - SECONDS
        for name in sources:
            audio = read_audio(VA / name)
            for start in (0, last // 2, last):
                clip = audio[start * SR:(start + SECONDS) * SR]
                for channel in ("original", BANDLIMIT, CODEC):
                    signal = clip if channel == "original" else phone_audio(clip, channel)
                    stream = inference.Stream(detector)
                    stream.feed(signal)
                    status = stream.finish()
                    assert status["duration_s"] == SECONDS
                    trials.append(dict(source_family=family, file=name, label=lookup[name]["label"],
                                       start_s=start, channel=channel, **status))
    return trials


def compare(trials):
    originals = {(r["source_family"], r["start_s"], r["channel"]): r
                 for r in trials if r["label"] == 0}
    summary = []
    for family in ["all", *sorted({r["source_family"] for r in trials})]:
        for channel in ["original", BANDLIMIT, CODEC]:
            rows = [r for r in trials if (family == "all" or r["source_family"] == family)
                    and r["channel"] == channel]
            fakes = [r for r in rows if r["label"]]
            reals = [r for r in rows if not r["label"]]
            gaps = []
            for r in fakes:
                ref = originals[r["source_family"], r["start_s"], channel]
                if r["score"] is not None and ref["score"] is not None:
                    gaps.append(r["score"] - ref["score"])
            summary.append(dict(family=family, channel=channel, fake_windows=len(fakes),
                                fake_flagged=sum(r["verdict"] == "FAKE" for r in fakes),
                                genuine_windows=len(reals),
                                genuine_flagged=sum(r["verdict"] == "FAKE" for r in reals),
                                insufficient=sum(r["verdict"] == "INSUFFICIENT" for r in rows),
                                comparable_pairs=len(gaps), fake_score_higher=sum(g > 0 for g in gaps),
                                median_fake_minus_original=float(np.median(gaps)) if gaps else None))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    detector = inference.Detector(device="cpu")
    audited = audit(detector.rule)
    trials = matched_trials(detector, audited)
    summary = compare(trials)
    with TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "results"
        stage.mkdir()
        write_csv(stage / "corpus_audit.csv", audited)
        write_csv(stage / "trials.csv", trials)
        write_csv(stage / "comparison.csv", summary)
        metadata = dict(seconds=SECONDS, positions=["start", "middle", "end"],
                        checkpoint_sha256=detector.rule["checkpoint_sha256"],
                        rule_sha256=checksum(RUN / "full/rule.json"),
                        script_sha256=checksum(Path(__file__)),
                        torch_version=str(torch.__version__),
                        ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0])
        (stage / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        stage.rename(args.out)
    for row in summary[:3]:
        print(row)


if __name__ == "__main__":
    main()
