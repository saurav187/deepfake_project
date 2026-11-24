"""Paired original/Opus experiment on a deterministic subset of existing v1 development IDs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .audio import SR, decode, ffmpeg
from .metrics import metrics


def run(repo, n):
    from detector.inference import load_detector

    dev = repo / "detector/runs/compare/x1_vs_v1/dev_scores.csv"
    with dev.open() as stream:
        ids = {r["uid"] for r in csv.DictReader(stream)}
    detector = load_detector()
    rng = np.random.default_rng(421)
    for src in ("asvspoof5", "indiefake"):
        with (repo / f"detector/runs/manifests/{src}.csv").open() as stream:
            rows = [
                r
                for r in csv.DictReader(stream)
                if r["uid"] in ids and 5 <= float(r["speech_s"]) <= 9
            ]
        groups = defaultdict(list)
        for r in rows:
            groups[r["label"], r["group"]].append(r)
        results = defaultdict(list)
        for lab in ("0", "1"):
            keys = sorted(k for k in groups if k[0] == lab)
            rng.shuffle(keys)
            used = 0
            while used < n:
                available = False
                for k in keys:
                    if not groups[k] or used >= n:
                        continue
                    available = True
                    r = groups[k].pop(rng.integers(len(groups[k])))
                    loc = r["locator"]
                    if src == "asvspoof5":
                        path, span = loc.removeprefix("tar:").split("::")
                        offset, length = map(int, span.split(":"))
                        if r["role"] != "dev" or Path(path).name != "flac_D_aa.tar":
                            raise ValueError("Unexpected ASV dev locator")
                        with Path(path).open("rb") as f:
                            f.seek(offset)
                            b = f.read(length)
                    else:
                        path = Path(loc.removeprefix("file:")).resolve()
                        if not path.is_relative_to(
                            (repo / "deepfake_model/datasets/indiefake").resolve()
                        ):
                            raise ValueError("Unexpected IndieFake locator")
                        b = path.read_bytes()
                    x = decode(b)
                    coded = ffmpeg(
                        x.astype("<f4").tobytes(),
                        ["-f", "f32le", "-ar", str(SR), "-ac", "1"],
                        ["-c:a", "libopus", "-b:a", "16k", "-f", "ogg"],
                    )
                    for name, audio in (("original", b), ("opus", coded)):
                        result = detector.analyze(audio, "development")
                        results[name].append(
                            (int(lab), result["fake_probability"], r["group"])
                        )
                    used += 1
                if not available:
                    break
        for view, scored in results.items():
            vals = [(y, s) for y, s, _ in scored if s is not None]
            print(
                json.dumps(
                    dict(
                        source=src,
                        view=view,
                        speakers=len({g for _, _, g in scored}),
                        eligible=len(vals),
                        total=len(scored),
                        median_real=float(np.median([s for y, s in vals if not y])),
                        median_fake=float(np.median([s for y, s in vals if y])),
                        **metrics([y for y, _ in vals], [s for _, s in vals]),
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--per-class", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(3)
    run(a.repo.resolve(), a.per_class)
