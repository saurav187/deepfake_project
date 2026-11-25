"""English XTTS-only shared treatment: crop cache, train-only diagnostic, frozen choice.

Owner Option A, 25 Nov 2026. The original failed audit of the untreated 4 s crops in
stage0-en-ecapa-xtts/ is preserved untouched. This module never reads a non-train row.

  cache     one source read: the original 4 s fixed crops of every selection row, written to
            a local directory and verified against the original eligible manifest's sha256
  diagnose  writes protocol.json (treatments, parameters, folds, rule) BEFORE any scoring,
            then compares the treatments on train only with reader-disjoint grouped CV
            (each genuine parent and its fakes in one fold) and writes frozen.json

Every treatment is applied identically to both labels and returns a 16-bit mono 16 kHz WAV
of exactly the 4 s window, so the model input path (audio.model_view) is unchanged.

  python -m pipeline_v3.en_treatment cache    --data $D --cache /content/crops
  python -m pipeline_v3.en_treatment diagnose --data $D --cache /content/crops
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BASE = "stage0-en-ecapa-xtts"  # the original selection, eligible manifest and failed audit
DIAG = "stage0-en-ecapa-xtts-diag"
TREATMENTS = ("norm", "norm_lp7k", "norm_opus")  # simplest first; the tie-break order
LEVEL_RMS = 0.05
PARAMS = {
    "input": "audio.fixed_crop output (4 s centre crop of the trimmed span), as audited",
    "norm": f"x * {LEVEL_RMS} / max(rms(x), 1e-6)  (unit level, fixed headroom)",
    "norm_lp7k": "norm, then scipy sosfiltfilt(butter(10, 7000 Hz, fs=16000, sos))",
    "norm_opus": "norm, then ffmpeg libopus 16 kb/s -application voip (Ogg), decoded back "
    "to 16 kHz mono; exactly WINDOW samples kept (fails closed if shorter)",
    "output": "16-bit PCM mono 16 kHz WAV, exactly WINDOW samples",
}
FOLDS = 5
TIE = 0.01
RULE = {
    "rows": "eligible.csv rows of the original selection with split == 'train' only",
    "folds": f"{FOLDS} reader-disjoint folds; every row is grouped by its genuine parent's "
    "reader (sha256('en-treat-cv-v1/'+reader) order, round-robin), so a parent and its "
    "fakes share a fold",
    "probe": "audit.fit_controls per fold (fit on 4 folds, score the held-out fold), the "
    "same descriptors, provenance fields excluded, joint probe included",
    "score": "per blocking field: mean over folds of max(AUC, 1-AUC), missing AUC = 1.0; "
    "treatment score = max over blocking fields (incl. joint)",
    "choice": f"lowest score; any treatment within {TIE} of the lowest is a tie and the "
    "simplest tied treatment (order norm < norm_lp7k < norm_opus) is frozen",
    "after": "one formal audit of the frozen treatment's exact bytes with the existing 0.60 "
    "gate on select; no runner-up trial",
}


def _wav(x):
    import io
    import wave

    from .audio import SR

    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setparams((1, 2, SR, 0, "NONE", ""))
        f.writeframes((np.clip(x, -1, 32767 / 32768) * 32768).round().astype("<i2").tobytes())
    return buf.getvalue()


def apply(crop, name):
    """Treated bytes from the audited crop bytes; identical for both labels."""
    import subprocess

    from .audio import WINDOW, decode

    if name not in TREATMENTS:
        raise ValueError("Unknown treatment " + name)
    x = decode(crop)
    if len(x) != WINDOW:
        raise ValueError("Crop is not exactly the 4 s window")
    x = LEVEL_RMS * x / max(float(np.sqrt(np.mean(x**2))), 1e-6)
    if name == "norm_lp7k":
        from scipy.signal import butter, sosfiltfilt

        x = sosfiltfilt(butter(10, 7000, fs=16000, output="sos"), x)
    elif name == "norm_opus":
        ogg = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000",
             "-c:a", "libopus", "-b:a", "16k", "-application", "voip", "-f", "ogg", "pipe:1"],
            input=_wav(x), capture_output=True, check=True, timeout=60,
        ).stdout
        x = decode(ogg)
        if len(x) < WINDOW:
            raise ValueError("Opus round trip shorter than the 4 s window")
        x = x[:WINDOW]
    return _wav(x)


def _h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def folds(rows, k=FOLDS):
    """Reader-disjoint fold per row id; a genuine parent and its fakes share a fold."""
    reader = {r["parent"]: r["speaker"] for r in rows if r["label"] == "0"}
    if any(r["parent"] not in reader for r in rows):
        raise ValueError("Fake without a genuine parent in the diagnostic rows")
    order = sorted(set(reader.values()), key=lambda s: _h("en-treat-cv-v1/" + s))
    fold_of = {s: i % k for i, s in enumerate(order)}
    return {r["id"]: fold_of[reader[r["parent"]]] for r in rows}


def score(per_fold):
    """per_fold: [{field: auc}] -> (treatment score, per-field mean oriented AUC)."""
    fields = sorted(set().union(*per_fold))
    mean = {
        f: float(np.mean([1.0 if d.get(f) is None else max(d[f], 1 - d[f]) for d in per_fold]))
        for f in fields
    }
    return max(mean.values()), mean


def choose(scores, tie=TIE):
    best = min(scores.values())
    return next(t for t in TREATMENTS if t in scores and scores[t] <= best + tie)


# ---- crop cache -------------------------------------------------------------------------


def _cache_archive(args):
    root, archive, items, cache = args
    from .adapters import selected_members
    from .audio import fixed_crop

    out = {}
    for r, b in selected_members({Path(archive): items}):
        c = fixed_crop(b)
        if c is None:
            out[r["id"]] = None
            continue
        sha = hashlib.sha256(c).hexdigest()
        path = Path(cache) / (r["id"] + ".wav")
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(c)
        tmp.replace(path)
        out[r["id"]] = sha
    return out


def cache_crops(data, cache):
    from .data import digest, safe_path
    from .stage0 import read

    root, base = data / "development", data / BASE
    original = json.loads((base / "prep-audit.json").read_text())
    if digest(base / "selection.csv") != original["selection_sha256"]:
        raise ValueError("Original selection changed since its audit")
    if digest(base / "eligible.csv") != original["eligible_sha256"]:
        raise ValueError("Original eligible manifest changed since its audit")
    rows = read(base / "selection.csv")
    if any(r["pair"] != "phonemedf" for r in rows):
        raise ValueError("Crop cache is only for PhonemeDF rows")
    cache.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for r in rows:
        grouped[str(safe_path(root, r["archive"]))].append(r)
    manifest = {}
    with ProcessPoolExecutor(len(grouped)) as ex:
        for part in ex.map(_cache_archive, [(root, a, v, str(cache)) for a, v in grouped.items()]):
            manifest.update(part)
    if set(manifest) != {r["id"] for r in rows}:
        raise ValueError("Crop cache does not cover the selection")
    verify(rows, read(base / "eligible.csv"), manifest)
    result = {
        "selection_sha256": original["selection_sha256"],
        "eligible_sha256": original["eligible_sha256"],
        "rows": len(manifest),
        "crops": sum(v is not None for v in manifest.values()),
        "too_short": sum(v is None for v in manifest.values()),
        "verified_against_eligible": True,
    }
    (cache / "manifest.json").write_text(json.dumps(manifest))
    (cache / "cache-verify.json").write_text(json.dumps(result, indent=2))
    return result


def verify(rows, eligible, manifest):
    """The cached crops reproduce the original audit's eligible set and hashes exactly."""
    available = [r for r in rows if manifest.get(r["id"])]
    joined = {r["parent"] for r in available if r["label"] == "0"} & {
        r["parent"] for r in available if r["label"] == "1"
    }
    if {r["id"] for r in available if r["parent"] in joined} != {r["id"] for r in eligible}:
        raise ValueError("Cached crops do not reproduce the eligible set")
    bad = [r["id"] for r in eligible if manifest[r["id"]] != r["sha256"]]
    if bad:
        raise ValueError(f"{len(bad)} cached crops differ from the audited sha256")


def load_cache(cache, data):
    manifest = json.loads((cache / "manifest.json").read_text())
    check = json.loads((cache / "cache-verify.json").read_text())
    original = json.loads((data / BASE / "prep-audit.json").read_text())
    if (
        not check.get("verified_against_eligible")
        or check["eligible_sha256"] != original["eligible_sha256"]
        or check["selection_sha256"] != original["selection_sha256"]
    ):
        raise ValueError("Crop cache is not verified against the original audit")
    return manifest


def cached(cache, manifest, rid):
    sha = manifest[rid]
    if sha is None:
        return None
    b = (cache / (rid + ".wav")).read_bytes()
    if hashlib.sha256(b).hexdigest() != sha:
        raise ValueError("Cached crop changed: " + rid)
    return b


# ---- diagnostic -------------------------------------------------------------------------


def _features(args):
    cache, sha, rid, name = args
    from .audit import PROVENANCE_FIELDS, descriptors

    b = (Path(cache) / (rid + ".wav")).read_bytes()
    if hashlib.sha256(b).hexdigest() != sha:
        raise ValueError("Cached crop changed: " + rid)
    d = descriptors(apply(b, name))
    return {k: v for k, v in d.items() if k not in PROVENANCE_FIELDS}


def diagnose(data, cache, workers=None):
    from .audit import fit_controls
    from .data import digest
    from .stage0 import read

    base, diag = data / BASE, data / DIAG
    manifest = load_cache(cache, data)
    eligible = read(base / "eligible.csv")
    rows = [r for r in eligible if r["split"] == "train"]
    if not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("Diagnostic must use train rows only")
    fold_of = folds(rows)
    diag.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment": "English XTTS-only shared treatment diagnostic (owner Option A)",
        "eligible_sha256": digest(base / "eligible.csv"),
        "treatments": list(TREATMENTS),
        "parameters": PARAMS,
        "rule": RULE,
        "train_rows": len(rows),
        "train_readers": len({r["speaker"] for r in rows if r["label"] == "0"}),
        "fold_sizes": {i: sum(v == i for v in fold_of.values()) for i in range(FOLDS)},
        "fold_readers": {
            i: len({r["speaker"] for r in rows if r["label"] == "0" and fold_of[r["id"]] == i})
            for i in range(FOLDS)
        },
        "fold_assignment_sha256": _h(json.dumps(sorted(fold_of.items()))),
    }
    if (diag / "frozen.json").exists():
        raise ValueError("A treatment is already frozen; the diagnostic runs once")
    (diag / "protocol.json").write_text(json.dumps(protocol, indent=2))
    protocol_sha = digest(diag / "protocol.json")
    results = {}
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    for name in TREATMENTS:
        with ProcessPoolExecutor(workers) as ex:
            feats = list(
                ex.map(
                    _features,
                    [(str(cache), manifest[r["id"]], r["id"], name) for r in rows],
                    chunksize=32,
                )
            )
        per_fold = []
        for k in range(FOLDS):
            split = [dict(r, split="select" if fold_of[r["id"]] == k else "train") for r in rows]
            rep = fit_controls(split, feats)
            per_fold.append({f: rep["controls"][f]["auc"] for f in rep["blocking_fields"]})
        total, mean = score(per_fold)
        results[name] = {"score": total, "mean_oriented_auc": mean, "per_fold_auc": per_fold}
        (diag / "diagnostic.partial.json").write_text(json.dumps(results, indent=2))
    chosen = choose({t: v["score"] for t, v in results.items()})
    (diag / "diagnostic.json").write_text(
        json.dumps({"protocol_sha256": protocol_sha, "results": results}, indent=2)
    )
    frozen = {
        "treatment": chosen,
        "parameters": {"common": PARAMS["norm"], chosen: PARAMS[chosen], "output": PARAMS["output"]},
        "scores": {t: v["score"] for t, v in results.items()},
        "protocol_sha256": protocol_sha,
        "diagnostic_sha256": digest(diag / "diagnostic.json"),
        "select_rows_read": 0,
    }
    (diag / "frozen.json").write_text(json.dumps(frozen, indent=2))
    return frozen


def frozen_treatment(data):
    from .data import digest

    diag = data / DIAG
    frozen = json.loads((diag / "frozen.json").read_text())
    if (
        frozen["treatment"] not in TREATMENTS
        or digest(diag / "protocol.json") != frozen["protocol_sha256"]
        or digest(diag / "diagnostic.json") != frozen["diagnostic_sha256"]
    ):
        raise ValueError("Frozen treatment record does not match its diagnostic")
    return frozen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["cache", "diagnose"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--workers", type=int)
    a = p.parse_args()
    out = cache_crops(a.data, a.cache) if a.action == "cache" else diagnose(a.data, a.cache, a.workers)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
