"""Replay publisher matches and account for every English fake using metadata alone."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .match import select


def summary(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[
            "/".join(r[k] for k in ("language", "split", "generator", "label"))
        ].append(r)
    return {
        k: [
            len(v),
            len({r["speaker"] for r in v}),
            len(
                {
                    r["voice"]
                    for r in v
                    if r["voice"] != "-" and "/unknown:" not in r["voice"]
                }
            ),
        ]
        for k, v in sorted(cells.items())
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--fates", type=Path, required=True)
    args = p.parse_args()
    with args.inventory.open() as f:
        records = [json.loads(line) for line in f]
    with args.selection.open() as f:
        saved = list(csv.DictReader(f))
    fakes = [r for r in records if r["source"] == "mlaad"]
    details = {
        r["archive"]: {
            "archive": r["archive"],
            "family": Path(r["archive"]).parent.name,
            "original": r["metadata"]["original_file"],
            "voice": r["metadata"].get("reference_speaker", ""),
        }
        for r in fakes
    }
    print("inventory", Counter(r["source"] for r in records))
    print(
        "original readers",
        Counter(
            r["member"].split("/by_book/")[-1].split("/")[1]
            for r in records
            if r["source"] == "mailabs"
        ),
    )
    for mode in ("legacy", "named", "cohort"):
        fate = {}
        rows, rejected, _ = select(
            records, ["gemini-3.1-flash-tts", "freevc24"], fate=fate, voice_rule=mode
        )
        if mode == "legacy" and {r["id"]: r for r in rows} != {
            r["id"]: r for r in saved
        }:
            raise ValueError("Replay differs from saved selection")
        if fate.keys() != details.keys():
            raise ValueError("Incomplete fate ledger")
        for k, v in fate.items():
            details[k][mode] = v
        print(mode, json.dumps(summary(rows), sort_keys=True), rejected)
        print("fates", dict(Counter(fate.values())))
    uk = sorted(
        {
            r["metadata"]["original_file"]
            for r in fakes
            if r["metadata"]["original_file"].startswith("en_UK/")
        }
    )
    hypothetical = [
        {
            "source": "mailabs",
            "archive": "mailabs/en_UK.tgz",
            "member": name,
            "metadata": {},
        }
        for name in uk
    ]
    rows, _, _ = select(
        records + hypothetical, ["gemini-3.1-flash-tts", "freevc24"], voice_rule="named"
    )
    print(
        "UK projection assuming each referenced file exists", json.dumps(summary(rows))
    )
    print(
        "primary before acquisition",
        json.dumps(summary([r for r in saved if r["language"] == "hi"])),
    )
    with args.fates.open("x", newline="") as f:
        writer = csv.DictWriter(f, next(iter(details.values())).keys())
        writer.writeheader()
        writer.writerows(details.values())


if __name__ == "__main__":
    main()
