"""Stage 1 Hindi codec inspection (owner Option A, step 4). Development data only.

Takes matched Hindi genuine/fake pairs from the base selection's train and select folds,
records what can be verified about each stored payload (container, codec, sample rate,
bit rate, channels, duration), then measures the acoustic shortcut controls under three
treatments applied identically to both labels:

  baseline  the stored bytes as decoded by the pipeline
  aac       decode -> mono 16 kHz -> AAC-LC 64 kb/s (ADTS) -> pipeline decode
  opus      decode -> mono 16 kHz -> Opus 16 kb/s VoIP (Ogg), WhatsApp-like -> pipeline decode
  crop                  the PhonemeDF fixed 4 s centre crop of the trimmed span (audio.fixed_crop)
  crop_norm             crop, then unit-RMS level as the model input normalises it
  crop_norm_lp7k        crop_norm + shared 7 kHz low-pass (bandwidth equalisation)
  crop_norm_opus        crop_norm + Opus 16 kb/s VoIP
  active_norm           trimmed span, the 4 s window with the MOST active 10 ms frames (same
                        rule for both labels), unit level: tests whether speech_ratio is a
                        window-placement artefact
The crop_* treatments approximate what the model receives; the first three describe storage.

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


MODEL_VIEW = ("crop", "crop_norm", "crop_norm_lp7k", "crop_norm_opus")


def _wav(x):
    import io
    import wave

    import numpy as np

    from .audio import SR

    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setparams((1, 2, SR, 0, "NONE", ""))
        f.writeframes((np.clip(x, -1, 32767 / 32768) * 32768).round().astype("<i2").tobytes())
    return buf.getvalue()


def _active_window(x):
    """Trim like the pipeline, then take the 4 s window with the most active 10 ms frames."""
    import numpy as np

    from .audio import WINDOW, prepare

    t, _ = prepare(x)
    if len(t) < WINDOW:
        raise ValueError("shorter than the 4 s crop")
    hop = 160
    n = len(t) // hop
    frames = np.sqrt(np.mean(t[: n * hop].reshape(n, hop) ** 2, axis=1))
    active = (frames > max(0.001, 0.03 * np.max(np.abs(t)))).astype(float)
    w = WINDOW // hop
    if n <= w:
        return t[:WINDOW]
    best = int(np.argmax(np.convolve(active, np.ones(w), "valid")))
    return t[best * hop : best * hop + WINDOW]


def treat(b, name):
    if name == "baseline":
        return b
    if name == "active_norm":
        import numpy as np

        from .audio import decode

        x = _active_window(decode(b))
        return _wav(0.05 * x / max(float(np.sqrt(np.mean(x**2))), 1e-6))
    if name in MODEL_VIEW:
        import numpy as np

        from .audio import decode, fixed_crop

        c = fixed_crop(b)
        if c is None:
            raise ValueError("shorter than the 4 s crop")
        if name == "crop":
            return c
        x = decode(c)
        x = 0.05 * x / max(float(np.sqrt(np.mean(x**2))), 1e-6)  # unit level, fixed headroom
        if name == "crop_norm_lp7k":
            from scipy.signal import butter, sosfiltfilt

            x = sosfiltfilt(butter(10, 7000, fs=16000, output="sos"), x)
        y = _wav(x)
        return treat(y, "opus") if name == "crop_norm_opus" else y
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


ALL_TREATMENTS = ("baseline", *TREATMENTS, *MODEL_VIEW, "active_norm")


def run(data, per_split, base, treatments=ALL_TREATMENTS):
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
    for name in treatments:
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
    p.add_argument("--treatments", nargs="+", default=list(ALL_TREATMENTS), choices=ALL_TREATMENTS)
    a = p.parse_args()
    result = run(a.data, a.pairs, a.base, a.treatments)
    out = a.data / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
