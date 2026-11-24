"""Matched English publisher panel for descriptive scoring outside all decision folds."""

import json
from collections import Counter
from pathlib import Path


def mlaad_panel(inventory):
    reals = {}
    fakes = []
    with inventory.open() as f:
        for line in f:
            r = json.loads(line)
            if r["source"] == "mailabs":
                member = r["member"]
                offset = member.find("en_US/by_book/")
                if offset >= 0:
                    key = member[offset:]
                    if key in reals:
                        raise ValueError("Duplicate M-AILABS path")
                    reals[key] = r
            elif r["source"] == "mlaad":
                fakes.append(r)
    rows = []
    for fake in fakes:
        m = fake["metadata"]
        real = reals.get(m["original_file"])
        if not real:
            continue
        rows.append(
            {
                "family": Path(fake["archive"]).parent.name,
                "original": m["original_file"],
                "real_archive": real["archive"],
                "real_member": real["member"],
                "fake_archive": fake["archive"],
                "fake_member": fake["member"],
                "voice": m.get("reference_speaker") or "unknown",
            }
        )
    print(
        "Separate MLAAD diagnostic, matched fakes", Counter(r["family"] for r in rows)
    )
    return rows
