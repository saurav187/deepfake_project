"""Pair development prompts and consent-reviewed voices within language and fold."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .data import SPLITS, safe_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--prompts", required=True)
    p.add_argument("--voices", required=True)
    p.add_argument("--out", default="jobs.csv")
    p.add_argument("--per-voice", type=int, default=40)
    p.add_argument("--dry-run", action="store_true")
    cfg = p.parse_args()
    root = cfg.data / "development"

    def read(name):
        with safe_path(root, name).open() as f:
            return list(csv.DictReader(f))

    voices, prompts = read(cfg.voices), read(cfg.prompts)
    assigned = {}
    for rows, keys in (
        (voices, ("speaker", "voice")),
        (prompts, ("text_family", "text")),
    ):
        for row in rows:
            if row["split"] not in SPLITS or row["language"] not in {"en", "hi"}:
                raise ValueError("Invalid development partition or language")
            for key in keys:
                value = row[key].strip().casefold()
                if (
                    not value
                    or assigned.setdefault((key, value), row["split"]) != row["split"]
                ):
                    raise ValueError("Missing identity or split overlap")
    jobs = []
    for voice in voices:
        eligible = [
            r
            for r in prompts
            if (r["language"], r["split"]) == (voice["language"], voice["split"])
        ]
        if len(eligible) < cfg.per_voice:
            raise ValueError("Insufficient distinct prompts for voice")
        for r in eligible[: cfg.per_voice]:
            j = dict(voice, **{k: r[k] for k in ("text", "text_family", "seconds")})
            j["id"] = hashlib.sha256(
                json.dumps(j, sort_keys=True).encode()
            ).hexdigest()[:20]
            jobs.append(j)
    if not jobs or len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Empty or duplicated synthesis plan")
    from collections import Counter

    print(json.dumps(Counter(j["engine"] for j in jobs), indent=2))
    if not cfg.dry_run:
        with safe_path(root, cfg.out).open("x", newline="") as f:
            w = csv.DictWriter(f, jobs[0].keys())
            w.writeheader()
            w.writerows(jobs)


if __name__ == "__main__":
    main()
