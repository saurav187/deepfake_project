"""Stage 1 Hindi codec inspection (owner Option A, step 4). Development data only.

Takes matched Hindi genuine/fake pairs from the base selection's train and select folds,
records what can be verified about each stored payload (container, codec, sample rate,
bit rate, channels, duration), then measures the acoustic shortcut controls under three
treatments applied identically to both labels:

  baseline  the stored bytes as decoded by the pipeline
  aac       decode -> mono 16 kHz -> AAC-LC 64 kb/s (ADTS) -> pipeline decode
  opus      decode -> mono 16 kHz -> Opus 16 kb/s VoIP (Ogg), WhatsApp-like -> pipeline decode

Pair IDs and folds are preserved; probes are fitted on train and scored on select.
Shared re-encoding does NOT prove that earlier compression traces disappeared; this
reports probe AUCs only. Detector false alarms and recall belong to the pilot.

  python -m pipeline_v3.codec_probe --data /content/drive/MyDrive/audio-v3 --pairs 150
"""

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .adapters import selected_members
from .audit import PROVENANCE_FIELDS, descriptors, fit_controls
from .data import safe_path

TREATMENTS = {
    "aac": ["-c:a", "aac", "-b:a", "64k", "-f", "adts"],
    "opus": ["-c:a", "libopus", "-b:a", "16k", "-application", "voip", "-f", "ogg"],
}


def probe(b):
    cmd = [
        "ffprobe", "-v", "error", "-protocol_whitelist", "pipe",
        "-show_entries",
        "format=format_name,bit_rate,duration:stream=codec_name,sample_rate,channels,bit_rate",
        "-of", "json", "pipe:0",
    ]
    out = json.loads(subprocess.run(cmd, input=b, capture_output=True, check=True, timeout=30).stdout)
    s = (out.get("streams") or [{}])[0]
    f = out.get("format", {})
    return {
        "container": f.get("format_name"),
        "codec": s.get("codec_name"),
        "sample_rate": s.get("sample_rate"),
        "channels": s.get("channels"),
        "bit_rate": s.get("bit_rate") or f.get("bit_rate"),
        "duration": float(f["duration"]) if f.get("duration") else None,
    }


def treat(b, name):
    if name == "baseline":
        return b
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000",
         *TREATMENTS[name], "pipe:1"],
        input=b, capture_output=True, check=True, timeout=60,
    ).stdout


def choose(rows, per_split):
    """Matched (real, fake) pairs per fold, deterministic, at most one fake per parent."""
    hi = [r for r in rows if r["language"] == "hi" and r["split"] in {"train", "select"}]
    reals = {r["parent"]: r for r in hi if r["label"] == "0"}
    picked = defaultdict(list)
    seen = set()
    for r in sorted(hi, key=lambda r: r["id"]):
        if r["label"] != "1" or r["parent"] in seen or r["parent"] not in reals:
            continue
        if len(picked[r["split"]]) >= per_split:
            continue
        seen.add(r["parent"])
        picked[r["split"]].append((reals[r["parent"]], r))
    return [x for split in ("train", "select") for pair in picked[split] for x in pair]


def run(data, per_split, base):
    root = data / "development"
    with (root / base).open() as f:
        rows = choose(list(csv.DictReader(f)), per_split)
    groups = defaultdict(list)
    for r in rows:
        groups[safe_path(root, r["archive"])].append(r)
    payloads = {r["id"]: b for r, b in selected_members(groups)}
    stored = []
    for r in rows:
        meta = probe(payloads[r["id"]])
        stored.append({"label": r["label"], "source": r["source"], "generator": r["generator"], **meta})
    summary = {
        "pairs_per_split": Counter(r["split"] for r in rows if r["label"] == "1"),
        "generators": Counter(r["generator"] for r in rows if r["label"] == "1"),
        "distinct_real_readers": {
            s: len({r["speaker"] for r in rows if r["split"] == s and r["label"] == "0"})
            for s in ("train", "select")
        },
        "stored_payloads": {
            label: Counter(
                f"{m['container']}/{m['codec']}/{m['sample_rate']}Hz/{m['channels']}ch"
                for m in stored if m["label"] == label
            )
            for label in ("0", "1")
        },
        "stored_bit_rate_kbps": {
            label: sorted(
                round(int(m["bit_rate"]) / 1000)
                for m in stored if m["label"] == label and m["bit_rate"]
            )[:: max(1, per_split // 5)]
            for label in ("0", "1")
        },
        "treatments": {},
        "notes": [
            "Shared re-encoding does not prove earlier compression traces disappeared.",
            "Probe AUCs only; detector false alarms and recall are measured in the pilot.",
            "Provenance fields (source/pair) are excluded from these acoustic panels.",
        ],
    }
    for name in ("baseline", *TREATMENTS):
        feats, kept = [], []
        for r in rows:
            try:
                d = descriptors(treat(payloads[r["id"]], name))
            except (subprocess.SubprocessError, ValueError):
                continue
            feats.append({k: v for k, v in d.items() if k not in PROVENANCE_FIELDS})
            kept.append(r)
        report = fit_controls(kept, feats)
        summary["treatments"][name] = {
            "rows": len(kept),
            "pass": report["pass"],
            "failed_fields": report["failed_fields"],
            "auc": {k: v["auc"] for k, v in report["controls"].items()},
        }
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=150, help="matched pairs per fold (train, select)")
    p.add_argument("--base", default="selection_v0_english_thin.csv")
    p.add_argument("--out", default="stage0/codec_probe_hi.json")
    a = p.parse_args()
    result = run(a.data, a.pairs, a.base)
    out = a.data / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
