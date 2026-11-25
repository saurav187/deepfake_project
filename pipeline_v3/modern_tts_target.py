"""Measure detection and false-alarm tradeoffs on clean MLAAD development audio."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from .v2a import V1, save

FAMILIES = {"Chatterbox", "Gemini-3.1-Flash-TTS", "ElevenLabs-v3", "FishTTS"}


def point(samples, required):
    return sorted(samples, reverse=True)[required - 1]


def describe(groups):
    fakes = {family: groups[family] for family in sorted(FAMILIES)}
    real = groups["genuine"]
    pooled = [s for values in fakes.values() for s in values]
    if (
        not real
        or not all(fakes.values())
        or not all(map(math.isfinite, real + pooled))
    ):
        raise ValueError("Incomplete or nonfinite panel")
    thresholds = {
        "at_15_percent_false_alarms": math.nextafter(
            point(real, len(real) * 15 // 100 + 1), math.inf
        ),
        "pooled_at_least_70": point(pooled, (len(pooled) * 70 + 99) // 100),
        "each_family_at_least_70": min(
            point(scores, (len(scores) * 70 + 99) // 100) for scores in fakes.values()
        ),
    }
    out = {}
    for name, threshold in thresholds.items():
        out[name] = {
            "threshold": threshold,
            "false_alarms": sum(s >= threshold for s in real),
            "genuine": len(real),
            "detected_by_family": {
                k: [sum(s >= threshold for s in v), len(v)] for k, v in fakes.items()
            },
        }
    return out


def run(root):
    work = root / "v2/day-target-1"
    frozen = json.loads((work / "frozen.json").read_text())
    models = defaultdict(lambda: defaultdict(list))
    with (root / V1 / "runs/teacher-eval-clips.csv").open() as f:
        for r in csv.DictReader(f):
            if r["panel"] != "mlaad" or r["channel"] != "clean":
                continue
            family = (
                r["generator"].removeprefix("mlaad/")
                if r["label"] == "1"
                else "genuine"
            )
            if family not in FAMILIES | {"genuine"}:
                raise ValueError(f"Unexpected family {family}")
            models["v1"][family].append(float(r["p"]))
            key = hashlib.sha256(r["id"].encode()).hexdigest()
            scores = json.loads((work / "scores" / (key + ".json")).read_text())[
                "clean"
            ]
            for head in frozen["heads"]:
                models[head][family].append(scores[head])
            fallback = work / "fallback-detector-v1" / "scores" / (key + ".json")
            if fallback.exists():
                # Released three-head detector v1 (bdb7748c), paired view: same 4 s crop,
                # norm and channel as the other candidates; its frozen cutoff is unchanged.
                models["fallback_release_v1"][family].append(
                    json.loads(fallback.read_text())["clean"]["paired"]["p"]
                )
    report = {
        "scope": "In-sample clean development threshold sensitivity; frozen decisions unchanged",
        "models": {
            k: describe(v)
            for k, v in models.items()
            if k != "fallback_release_v1" or sum(map(len, v.values())) == sum(map(len, models["v1"].values()))
        },
    }
    save(work / "modern-tts-70-at-15-development.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    print(json.dumps(run(p.parse_args().data), indent=2))
