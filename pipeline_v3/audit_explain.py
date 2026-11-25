"""Explain the failed English XTTS-only formal audit (frozen `norm`). CPU only, read-only.

Owner-approved analysis (25 Nov 2026). Nothing here changes a gate, a treatment, a fold or
a selection, and nothing can authorize catalog, pack or training: outputs go to
stage0-en-ecapa-xtts-norm/explain/ and the audit report is never rewritten.

Recovers per-clip descriptors of the exact audited bytes (cached original crops -> frozen
`norm`, each verified against the treated eligible.csv sha256), refits the audit's
train-fitted probes exactly as audit.fit_controls does (same rows, features, order), checks
that the overall select AUCs reproduce prep-audit.json, then scores select subsets:
XTTS vs its own genuine parents, Chatterbox vs its own genuine parents, and both again on
the parents shared by the two generators.

  python -m pipeline_v3.en_treatment cache --data $D --cache /content/crops
  python -m pipeline_v3.audit_explain --data $D --cache /content/crops
"""

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

WORK = "stage0-en-ecapa-xtts-norm"
TREATMENT = "norm"
FIELDS = ("upper_band", "rolloff_hz", "tilt", "speech_ratio", "joint")


def _features(args):
    cache, crop_sha, treated_sha, rid, source, pair = args
    from .audit import descriptors
    from .en_treatment import apply

    crop = (Path(cache) / (rid + ".wav")).read_bytes()
    if hashlib.sha256(crop).hexdigest() != crop_sha:
        raise ValueError("Cached crop changed: " + rid)
    b = apply(crop, TREATMENT)
    if hashlib.sha256(b).hexdigest() != treated_sha:
        raise ValueError("Treated bytes differ from the audited bytes: " + rid)
    return dict(descriptors(b), source=source, pair=pair)


def subset_auc(y, scores, idx):
    from .metrics import metrics

    idx = np.asarray(idx)
    return metrics(y[idx], scores[idx])["auc"]


def comparison(rows, y, scores, gen, parents=None):
    """Select fakes of `gen` vs their own genuine parents (optionally only `parents`)."""
    fakes = [
        i for i, r in enumerate(rows)
        if r["label"] == "1" and r["generator"] == gen and (parents is None or r["parent"] in parents)
    ]
    own = {rows[i]["parent"] for i in fakes}
    reals = [i for i, r in enumerate(rows) if r["label"] == "0" and r["parent"] in own]
    idx = fakes + reals
    auc = {f: subset_auc(y, scores[f], idx) for f in FIELDS}
    return {
        "fakes": len(fakes),
        "genuine_parents": len(reals),
        "readers": len({rows[i]["speaker"] for i in reals}),
        "target_voices": sorted({rows[i]["voice"] for i in fakes if rows[i].get("voice")}),
        "auc": auc,
        "oriented_auc": {f: None if a is None else max(a, 1 - a) for f, a in auc.items()},
    }


def explain(data, cache, workers=None):
    from .audit import probe_scores
    from .data import digest
    from .en_treatment import load_cache
    from .metrics import metrics
    from .stage0 import read

    work = data / WORK
    report = json.loads((work / "prep-audit.json").read_text())
    if digest(work / "eligible.csv") != report["eligible_sha256"]:
        raise ValueError("Treated eligible manifest changed since the audit")
    if (report.get("treatment") or {}).get("treatment") != TREATMENT:
        raise ValueError("Not the frozen-norm audit")
    manifest = load_cache(cache, data)
    rows = [r for r in read(work / "eligible.csv") if r["split"] in {"train", "select"}]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    import time

    t0, feats = time.time(), []
    with ProcessPoolExecutor(workers) as ex:
        for i, f in enumerate(ex.map(
            _features,
            [(str(cache), manifest[r["id"]], r["sha256"], r["id"], r["source"], r["pair"]) for r in rows],
            chunksize=32,
        ), 1):
            feats.append(f)
            if i % 500 == 0 or i == len(rows):
                rate = i / (time.time() - t0)
                print(f"PROGRESS {i}/{len(rows)} clips {rate:.1f}/s eta {(len(rows) - i) / rate / 60:.1f} min", flush=True)
    raw = probe_scores(rows, feats)
    sel = [r for r in rows if r["split"] != "train"]
    y = np.array([int(r["label"]) for r in sel])
    scores = {f: np.asarray(v) for f, v in raw.items()}
    reproduced = {f: metrics(y, s)["auc"] for f, s in scores.items()}
    mismatch = {
        f: (reproduced[f], report["controls"][f]["auc"])
        for f in reproduced
        if f in report["controls"] and abs(reproduced[f] - report["controls"][f]["auc"]) > 1e-9
    }
    if mismatch:
        raise ValueError(f"Refitted probes do not reproduce the audit: {mismatch}")

    def parents(gen):
        return {r["parent"] for r in sel if r["label"] == "1" and r["generator"] == gen}

    shared = parents("xtts_v2") & parents("chatterboxtts")
    result = {
        "kind": "EXPLANATION of the failed frozen-norm formal audit (not a gate, not a retry)",
        "audit_eligible_sha256": report["eligible_sha256"],
        "reproduced_overall_select_auc": {f: reproduced[f] for f in FIELDS},
        "reproduction_exact": True,
        "select_all": {"fakes": int(y.sum()), "genuine": int((1 - y).sum())},
        "xtts_vs_own_parents": comparison(sel, y, scores, "xtts_v2"),
        "chatterbox_vs_own_parents": comparison(sel, y, scores, "chatterboxtts"),
        "shared_parents": len(shared),
        "shared_parent_readers": len({r["speaker"] for r in sel if r["label"] == "0" and r["parent"] in shared}),
        "xtts_on_shared_parents": comparison(sel, y, scores, "xtts_v2", shared),
        "chatterbox_on_shared_parents": comparison(sel, y, scores, "chatterboxtts", shared),
        "train_descriptor_means": {
            k: {
                lab: float(np.mean([f[k] for f, r in zip(feats, rows) if r["split"] == "train" and r["label"] == lab]))
                for lab in "01"
            }
            for k in ("upper_band", "rolloff_hz", "tilt", "speech_ratio")
        },
        "select_descriptor_means": {
            k: {
                g: float(np.mean([f[k] for f, r in zip(feats, rows) if r["split"] != "train" and
                                  (r["generator"] if r["label"] == "1" else "genuine") == g]))
                for g in ("genuine", "xtts_v2", "chatterboxtts")
            }
            for k in ("upper_band", "rolloff_hz", "tilt", "speech_ratio")
        },
    }
    out = work / "explain"
    out.mkdir(exist_ok=True)
    keys = ["id", "split", "label", "generator", "parent", "speaker", "voice", "sha256"]
    names = [k for k in feats[0] if k not in ("source", "pair")]
    with (out / "descriptors.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + names)
        for r, d in zip(rows, feats):
            w.writerow([r.get(k, "") for k in keys] + [d[k] for k in names])
    with (out / "select_probe_scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "generator", "parent", *scores])
        for i, r in enumerate(sel):
            w.writerow([r["id"], r["label"], r["generator"], r["parent"], *(scores[k][i] for k in scores)])
    (out / "explain.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--workers", type=int)
    a = p.parse_args()
    print(json.dumps(explain(a.data, a.cache, a.workers), indent=2))


if __name__ == "__main__":
    main()
