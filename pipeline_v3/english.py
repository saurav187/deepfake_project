"""Build reproducible LibriSpeech transcript pairs for pilot or final development."""

import argparse
import csv
import json
import tarfile
from collections import defaultdict
from pathlib import Path

from .data import check, safe_path
from .english_report import summary
from .fetch import LIBRI
from .generate import generated_path
from .match import prompt, short, write_selection

# Readers per source subset. Allocate whole readers before assigning stock voices.
QUOTAS = {"train": 16, "select": 4, "calibrate": 4, "threshold": 3, "validate": 3}
VOICES = {
    "azure": [
        ("en-US-JennyNeural", "en-IN-NeerjaNeural"),
        ("en-US-GuyNeural", "en-GB-SoniaNeural"),
        ("en-US-AriaNeural", "en-IN-PrabhatNeural"),
        ("en-US-DavisNeural", "en-GB-RyanNeural"),
        ("en-US-JaneNeural", "en-US-JasonNeural"),
    ],
    "gemini": [
        ("Kore", "Puck"),
        ("Aoede", "Charon"),
        ("Leda", "Orus"),
        ("Zephyr", "Fenrir"),
        ("Sulafat", "Achernar"),
    ],
}


def transcripts(root, subset):
    archive = "librispeech/" + subset + ".tar.gz"
    members = set()
    prompts = []
    with tarfile.open(safe_path(root, archive), "r|gz") as tar:
        for m in tar:
            path = Path(m.name)
            if not m.isfile() or path.parts[:2] != ("LibriSpeech", subset):
                continue
            if path.suffix == ".flac":
                members.add(m.name)
            elif path.name.endswith(".trans.txt"):
                for line in tar.extractfile(m):
                    uid, text = line.decode().strip().split(" ", 1)
                    if uid.rsplit("-", 1)[0] != path.name.removesuffix(".trans.txt"):
                        raise ValueError("Unexpected LibriSpeech transcript ID")
                    if 6 <= len(text.split()) <= 45:
                        prompts.append(
                            (uid, text, archive, str(path.parent / (uid + ".flac")))
                        )
    if any(member not in members for _, _, _, member in prompts):
        raise ValueError("Missing LibriSpeech audio for transcript")
    return prompts


def plan(root, release=False):
    used_text = set()
    used_readers = set()
    jobs = []
    for subset in LIBRI:
        by_reader = defaultdict(list)
        for entry in sorted(transcripts(root, subset), key=lambda r: short(r[0])):
            by_reader[entry[0].split("-")[0]].append(entry)
        readers = iter(sorted(by_reader, key=lambda r: short("libri/" + r)))
        for fold, (split, quota) in enumerate(QUOTAS.items()):
            selected = 0
            for reader in readers:
                if reader in used_readers:
                    raise ValueError("Repeated reader across source subsets")
                lines = {}
                for entry in by_reader[reader]:
                    key = prompt(entry[1])
                    if key not in used_text:
                        lines.setdefault(key, entry)
                if len(lines) < 24:
                    continue
                used_readers.add(reader)
                for i, (key, (uid, text, archive, member)) in enumerate(
                    list(lines.items())[:24]
                ):
                    used_text.add(key)
                    engine = (
                        "gemini" if i % 2 and (split != "train" or release) else "azure"
                    )
                    voice = VOICES[engine][fold][
                        (selected + (subset == "dev-other")) % 2
                    ]
                    jobs.append(
                        {
                            "id": short(uid + "/" + engine + "/" + voice),
                            "text": text,
                            "engine": engine,
                            "model": "gemini-3.1-flash-tts-preview"
                            if engine == "gemini"
                            else "azure-neural",
                            "voice": voice,
                            "speaker": "librispeech/" + reader,
                            "language": "en",
                            "split": split,
                            "seconds": str(round(len(text.split()) / 2.5, 2)),
                            "archive": archive,
                            "member": member,
                        }
                    )
                selected += 1
                if selected == quota:
                    break
            if selected != quota:
                raise ValueError(
                    f"{subset}/{split} lacks eligible readers for {quota} x 24 pairs"
                )
    return jobs


def combine(hindi, jobs, release):
    rows = list(hindi)
    for job in jobs:
        original = job["archive"] + "/" + job["member"]
        parent = "librispeech/" + short(original)
        for fake in (False, True):
            archive = str(generated_path(job)) if fake else job["archive"]
            rows.append(
                {
                    "archive": archive,
                    "member": "-" if fake else job["member"],
                    "path": archive,
                    "id": short(str(generated_path(job)) if fake else original),
                    "speaker": job["speaker"],
                    "voice": job["engine"] + "/" + job["voice"].casefold()
                    if fake
                    else "-",
                    "label": str(int(fake)),
                    "language": "en",
                    "pair": "libri-stock",
                    "channel": "clean",
                    "generator": job["engine"] if fake else "-",
                    "split": job["split"],
                    "parent": parent,
                    "text": prompt(job["text"]),
                    "source": job["engine"] if fake else "librispeech",
                    "domain": (
                        "unseen"
                        if job["engine"] == "gemini" and not release
                        else "seen"
                    )
                    if fake
                    else "real",
                }
            )
    check(rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "assemble"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--base", default="selection.csv")
    p.add_argument("--release", action="store_true")
    args = p.parse_args()
    root = args.data / "development"
    version = "release" if args.release else "pilot"
    jobs = plan(root, args.release)
    with safe_path(root, args.base).open() as f:
        hindi = [r for r in csv.DictReader(f) if r["language"] == "hi"]
    if {r["split"] for r in hindi} != set(QUOTAS):
        raise ValueError("Missing Hindi development split")
    rows = combine(hindi, jobs, args.release)
    print(json.dumps(summary(rows), indent=2))
    if args.action == "plan":
        path = root / f"english-{version}-jobs.csv"
        if path.exists():
            with path.open() as f:
                if list(csv.DictReader(f)) == jobs:
                    return
            raise ValueError("Existing jobs differ from plan")
        with path.open("x", newline="") as f:
            writer = csv.DictWriter(f, jobs[0].keys())
            writer.writeheader()
            writer.writerows(jobs)
    else:
        for job in jobs:
            path = safe_path(root, generated_path(job))
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing generated job {job['id']}")
        write_selection(root / f"selection-{version}.csv", rows)


if __name__ == "__main__":
    main()
