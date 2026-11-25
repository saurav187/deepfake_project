"""Stage 1 Hindi speech-density matching experiment (one run, development data only).

Rule family: keep a matched (genuine parent, fake) pair only if the two clips' active-frame
ratios (audit `speech_ratio`, model view `crop_norm`) differ by at most DELTA. Pairs stay
intact, so parent, speaker, voice, text and generator boundaries are those of the base
selection. DELTA is chosen on TRAIN pairs only: the largest grid value whose train
speech_ratio AUC lies within 0.5 +/- 0.05 (a margin inside the 0.60 gate). It is then
frozen, applied once to SELECT, and every acoustic control is fitted on matched train and
scored on matched select with the unchanged 0.60 gate.

  python -m pipeline_v3.hindi_density --data /content/drive/MyDrive/audio-v3 --pairs 500
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .adapters import selected_members
from .audit import PROVENANCE_FIELDS, descriptors, fit_controls
from .codec_probe import choose, treat
from .data import safe_path

GRID = (0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01)
TREATMENT = "crop_norm"


def _auc(rows, feats):
    y = [int(r["label"]) for r in rows]
    return float(roc_auc_score(y, [f["speech_ratio"] for f in feats])) if len(set(y)) == 2 else None


def run(data, per_split, base):
    root = data / "development"
    with (root / base).open() as f:
        rows = choose(list(csv.DictReader(f)), per_split)
    groups = defaultdict(list)
    for r in rows:
        groups[safe_path(root, r["archive"])].append(r)
    feats = {}
    for r, b in selected_members(groups):
        try:
            d = descriptors(treat(b, TREATMENT))
        except Exception:  # noqa: BLE001 - short or undecodable clips are reported as lost
            continue
        feats[r["id"]] = {k: v for k, v in d.items() if k not in PROVENANCE_FIELDS}
    pairs = defaultdict(dict)
    for r in rows:
        if r["id"] in feats:
            pairs[r["split"], r["parent"]][r["label"]] = r
    pairs = {k: v for k, v in pairs.items() if set(v) == {"0", "1"}}

    def gap(p):
        return abs(feats[p["0"]["id"]]["speech_ratio"] - feats[p["1"]["id"]]["speech_ratio"])

    def keep(split, delta):
        return [p for (s, _), p in sorted(pairs.items()) if s == split and (delta is None or gap(p) <= delta)]

    def flat(ps):
        rs = [p[k] for p in ps for k in ("0", "1")]
        return rs, [feats[r["id"]] for r in rs]

    # Choose DELTA on train only.
    trace, delta = [], None
    for d in GRID:
        rs, fs = flat(keep("train", d))
        auc = _auc(rs, fs)
        trace.append({"delta": d, "train_pairs": len(rs) // 2, "train_speech_ratio_auc": auc})
        if delta is None and auc is not None and abs(auc - 0.5) <= 0.05:
            delta = d
    report = {"treatment": TREATMENT, "grid": trace, "frozen_delta": delta}

    def describe(ps):
        rs = [p[k] for p in ps for k in ("0", "1")]
        return {
            "pairs": len(ps),
            "generators": dict(Counter(p["1"]["generator"] for p in ps)),
            "real_readers": len({p["0"]["speaker"] for p in ps}),
            "fake_voices": len({p["1"]["voice"] for p in ps}),
            "texts": len({r["text"] for r in rs}),
        }

    def controls(train_ps, select_ps):
        rs, fs = flat(train_ps)
        rs2, fs2 = flat(select_ps)
        out = fit_controls(rs + rs2, fs + fs2)
        return {
            "pass": out["pass"],
            "failed_fields": out["failed_fields"],
            "auc": {k: v["auc"] for k, v in out["controls"].items()},
        }

    report["before"] = {
        "train": describe(keep("train", None)),
        "select": describe(keep("select", None)),
        "controls": controls(keep("train", None), keep("select", None)),
    }
    if delta is None:
        report["result"] = "no grid value balances train speech_ratio; Hindi blocked"
        return report
    tr, se = keep("train", delta), keep("select", delta)
    report["after"] = {
        "train": describe(tr),
        "select": describe(se),
        "coverage": {
            "train": len(tr) / max(1, len(keep("train", None))),
            "select": len(se) / max(1, len(keep("select", None))),
        },
        "controls": controls(tr, se) if tr and se else None,
    }
    report["note"] = (
        "Experiment on a deterministic pair sample. Passing here does not authorize Hindi: "
        "the pipeline would also need the same crop/level view and caliper applied in its "
        "own audit, with the resulting pack audited."
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=500)
    p.add_argument("--base", default="selection_v0_english_thin.csv")
    p.add_argument("--out", default="stage0/hindi_density.json")
    a = p.parse_args()
    result = run(a.data, a.pairs, a.base)
    out = a.data / a.out
    out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
