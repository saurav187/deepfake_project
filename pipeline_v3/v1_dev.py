"""Score fresh, explicitly development-only matched cohorts with the existing §7 backend."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import entries, payload
from .metrics import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    from detector.inference import load_detector

    detector = load_detector()
    groups = defaultdict(list)
    for r in entries(args.local, {"select"}):
        result = detector.analyze(payload(args.local, r), "dev.audio")
        for key in (
            "all",
            r["language"],
            r["pair"],
            r["generator"],
            r["channel"],
            r["source"],
        ):
            groups[key].append(
                (int(r["label"]), result["fake_probability"], result["verdict"])
            )
    report = {}
    for key, vals in groups.items():
        usable = [(y, s) for y, s, _ in vals if s is not None]
        report[key] = dict(
            **metrics([y for y, _ in usable], [s for _, s in usable]),
            total=len(vals),
            median={
                str(c): float(np.median(v))
                for c in (0, 1)
                if (v := [s for y, s in usable if y == c])
            },
            flagged=sum(d == "likely_fake" for _, _, d in vals),
        )
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
