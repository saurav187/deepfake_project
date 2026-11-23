"""Dev-only check: does the released model lean on how recordings start or end?

Usage (repo root):  venv/bin/python -m detector.eval.edge_reliance --release detector/runs/release/v1

ASVspoof 5 train had an edge shortcut (genuine files decay at the end, fakes stop
abruptly); the residual tail-level cue after prepare() was AUC 0.615 vs 0.537
mid-recording. For dev recordings with at least three 4 s windows, this compares
the logit of the first and last window with the mean of the interior windows.
If the model used endings as evidence of genuineness, edge windows of genuine
recordings would score clearly lower (more genuine) than their interior, and the
label would predict the edge-minus-interior difference. Report only; dev data.
"""
import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from detector import paths
from detector.audio import decode, prepare
from detector.data.sources import read_bytes
from detector.inference import Detector, windows_of


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--per-cell", type=int, default=500)
    args = p.parse_args()
    det = Detector(args.release)
    with (paths.AUDIT / "manifest.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["role"] == "dev" and not r["excluded"] and float(r["speech_s"] or 0) >= 12.5]
    cells = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["uid"]):
        cells[(r["source"], int(r["label"]))].append(r)
    rng = np.random.default_rng(5)
    sample = [items[i] for key, items in sorted(cells.items()) for i in rng.permutation(len(items))[:args.per_cell]]

    def load(row):
        try:
            return row, windows_of(prepare(decode(read_bytes(row["locator"]))[0]))[0]
        except Exception:
            return row, None

    out = defaultdict(lambda: {"first": [], "last": [], "label": []})
    with ThreadPoolExecutor(6) as pool:
        for row, w in pool.map(load, sample, buffersize=24):
            if w is None or len(w) < 3:
                continue
            l = det.window_logits(w)
            interior = l[1:-1].mean()
            d = out[row["source"]]
            d["first"].append(l[0] - interior)
            d["last"].append(l[-1] - interior)
            d["label"].append(int(row["label"]))
    report = {}
    for source, d in sorted(out.items()):
        y = np.array(d["label"])
        entry = dict(n=len(y), genuine=int((y == 0).sum()), fake=int((y == 1).sum()))
        for edge in ("first", "last"):
            v = np.array(d[edge])
            entry[f"{edge}_minus_interior_mean_genuine"] = float(v[y == 0].mean()) if (y == 0).any() else None
            entry[f"{edge}_minus_interior_mean_fake"] = float(v[y == 1].mean()) if (y == 1).any() else None
            if len(set(y)) == 2:
                a = roc_auc_score(y, v)
                entry[f"{edge}_shift_label_auc"] = float(max(a, 1 - a))
        report[source] = entry
        print(source, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in entry.items()}))
    (args.release / "edge_reliance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
