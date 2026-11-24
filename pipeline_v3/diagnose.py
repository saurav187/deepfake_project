"""Summarize only the saved v1/x1 development comparison, without fitting a model."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .metrics import metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    args = p.parse_args()
    if args.scores.name != "dev_scores.csv":
        raise ValueError("Expected development comparison CSV")
    with args.scores.open() as f:
        rows = list(csv.DictReader(f))
    if {r["source"] for r in rows} != {"asvspoof5", "indiefake"}:
        raise ValueError("Unexpected development sources")
    for col in rows[0]:
        if "|" not in col:
            continue
        for source in ("asvspoof5", "indiefake"):
            group = [
                r for r in rows if r["source"] == source and np.isfinite(float(r[col]))
            ]
            print(
                json.dumps(
                    dict(
                        model_view=col,
                        source=source,
                        **metrics(
                            [int(r["label"]) for r in group],
                            [float(r[col]) for r in group],
                        ),
                    )
                )
            )
            for rate, label in sorted({(r["rate"], r["label"]) for r in group}):
                values = [
                    float(r[col])
                    for r in group
                    if (r["rate"], r["label"]) == (rate, label)
                ]
                print(
                    json.dumps(
                        {
                            "source": source,
                            "model_view": col,
                            "rate": rate,
                            "label": label,
                            "n": len(values),
                            "median_logit": float(np.median(values)),
                        }
                    )
                )


if __name__ == "__main__":
    main()
