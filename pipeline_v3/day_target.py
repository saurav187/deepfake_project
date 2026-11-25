"""One-day English detector experiment. Select on select, freeze on threshold, compare on MLAAD."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import digest, entries, payload
from .metrics import metrics
from .v2a import V1, save

FAMILIES = ("Chatterbox", "ElevenLabs-v3", "FishTTS", "Gemini-3.1-Flash-TTS")
VIEWS = ("clean", "opus", "phone")
FPR = 0.02
TARGET = 0.80


def cutoff(genuine, budget=FPR):
    """Smallest representable cutoff above the first score that exceeds the FP allowance."""
    a = np.asarray(genuine, dtype=float)
    if a.size < 50 or not np.isfinite(a).all():
        raise ValueError("Need at least 50 finite genuine scores")
    boundary = np.sort(a)[::-1][math.floor(len(a) * budget)]
    return float(np.nextafter(boundary, np.inf))


def snapshot(work, name, record):
    path = work / name
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError(f"Frozen experiment input changed: {path}")
    save(path, record)


def reference(work):
    with (work / "runs/teacher-eval-clips.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["panel"] == "mlaad"]
    if len({(r["id"], r["channel"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate reference trials")
    return rows


def table(rows, threshold):
    cells = defaultdict(list)
    for r in rows:
        cells[r["channel"], r["generator"] if r["label"] == "1" else "genuine"].append(
            r
        )
    report = {}
    for (view, group), rr in sorted(cells.items()):
        hits = sum(float(r["p"]) >= threshold for r in rr)
        report[f"{view}/{group}"] = {
            "flagged": hits,
            "clips": len(rr),
            "rate": hits / len(rr),
            "readers": len({r["speaker"] for r in rr}),
        }
    required = {
        f"{v}/{g}"
        for v in VIEWS
        for g in ("genuine", *("mlaad/" + x for x in FAMILIES))
    }
    if set(report) != required:
        raise ValueError("Incomplete comparison panels")
    passed = all(
        x["rate"] <= FPR if k.endswith("/genuine") else x["rate"] >= TARGET
        for k, x in report.items()
    )
    return {"cells": report, "development_target_met": passed}


def fit_head(xs, ys):
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=400, class_weight="balanced")
    )
    model.fit(xs, ys)
    scaler, clf = model.steps[0][1], model.steps[1][1]
    w = clf.coef_[0] / scaler.scale_
    return {"weights": w.tolist(), "bias": float(clf.intercept_[0] - w @ scaler.mean_)}


def predict(head, x):
    return float(np.asarray(head["weights"]) @ x + head["bias"])


def vector(h):
    if h.ndim != 2 or len(h) < 2 or not np.isfinite(h).all():
        raise ValueError("Invalid encoder states")
    return h[1:].mean(axis=0, dtype=np.float32)


class Encoder:
    def __init__(self, path):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = (
            AutoModel.from_pretrained(str(path), local_files_only=True)
            .eval()
            .to(self.device)
        )

    def feature(self, raw, view):
        from .audio import channel, model_view

        x = model_view(channel(raw, view), {"pair": "phonemedf"})
        if len(x) != 64000:
            raise ValueError("Unexpected model input duration")
        with self.torch.inference_mode():
            h = self.model(
                self.torch.from_numpy(x).to(self.device)[None],
                output_hidden_states=True,
            ).hidden_states
            pooled = self.torch.stack(
                [self.torch.cat([v[0].mean(0), v[0].std(0, unbiased=False)]) for v in h]
            )
        return vector(pooled.cpu().numpy().astype(np.float16))


def fit(root, work):
    from .audio import decode
    from .cli import bound_audit
    from .pilot import cache_directory, sample_rows

    v1 = root / V1
    cfg = SimpleNamespace(
        data=v1, local=v1 / "packed", languages=["en"], seed=42, per_class=1000
    )
    idx = cfg.local / "index.json"
    audit = bound_audit(v1, idx)
    pilot = json.loads((v1 / "pilot.json").read_text())
    if (
        not pilot["go"]
        or pilot["index_sha256"] != digest(idx)
        or pilot["exception"] != audit["exception"]
    ):
        raise ValueError("Invalid original pilot")
    path = root / "models/wavlm-base"
    inputs = {
        "index": digest(idx),
        "pilot": digest(v1 / "pilot.json"),
        "encoder": encoder_id(path),
        "goal": TARGET,
        "max_false_alarm": FPR,
        "version": 1,
        "exception": audit["exception"],
    }
    snapshot(work, "protocol.json", inputs)
    out = work / "frozen.json"
    if out.exists():
        if json.loads(out.read_text())["protocol"] != inputs:
            raise ValueError("Stale frozen detector")
        return
    rows, _, _ = sample_rows(cfg)
    cache = cache_directory(cfg, path, "small")
    train = [r for r in rows if r["split"] == "train"]
    select = [r for r in rows if r["split"] == "select"]

    def cached(r, view):
        with np.load(cache / f"{r['id']}-{view}.npz", allow_pickle=False) as npz:
            return vector(npz["h"])

    # Same 2,000 labels and full-layer representation as the winning clean pilot.
    clean = [cached(r, "clean") for r in train]
    clean_y = [int(r["label"]) for r in train]
    heads = {
        "clean": fit_head(clean, clean_y),
        "clean_opus": fit_head(clean + [cached(r, "opus") for r in train], clean_y * 2),
    }
    groups = {}
    for view in ("clean", "opus"):
        feats = [cached(r, view) for r in select]
        for name, head in heads.items():
            groups[name, view] = np.array([predict(head, x) for x in feats])
    unseen = [
        i for i, r in enumerate(select) if r["label"] == "0" or r["domain"] == "unseen"
    ]
    replay = metrics(
        [int(select[i]["label"]) for i in unseen], groups["clean", "clean"][unseen]
    )
    saved = pilot["models"]["small/full/linear/clean"]["slices"]["domain/unseen"]
    if replay["n"] != saved["n"] or abs(replay["eer"] - saved["eer"]) > 1e-6:
        raise ValueError("Original pilot EER did not reproduce")

    def dev_rates(name, cut):
        rates = []
        for view in ("clean", "opus"):
            scores = groups[name, view]
            for gen in ("xtts_v2", "chatterboxtts"):
                rates.append(
                    float(
                        np.mean(
                            [
                                s >= cut
                                for r, s in zip(select, scores)
                                if r["generator"] == gen
                            ]
                        )
                    )
                )
        return rates

    def reference_cuts(split, candidates):
        net = Encoder(path)
        genuine = [r for r in entries(cfg.local, {split}) if r["label"] == "0"]
        scores = defaultdict(list)
        for r in genuine:
            raw = decode(payload(cfg.local, r))
            for view in VIEWS:
                x = net.feature(raw, view)
                for name in candidates:
                    scores[name, view].append(predict(heads[name], x))
        return {
            name: max(cutoff(scores[name, v]) for v in VIEWS) for name in candidates
        }

    cuts = reference_cuts("calibrate", heads)
    selection = {}
    for name in heads:
        fp = max(
            float(
                np.mean(
                    [
                        s >= cuts[name]
                        for r, s in zip(select, groups[name, v])
                        if r["label"] == "0"
                    ]
                )
            )
            for v in ("clean", "opus")
        )
        selection[name] = {
            "fpr": fp,
            "family_view_detection": dev_rates(name, cuts[name]),
        }

    def priority(name):
        p = selection[name]
        return (
            p["fpr"] <= FPR,
            -max(0.0, p["fpr"] - FPR),
            min(p["family_view_detection"]),
            np.mean(p["family_view_detection"]),
            name == "clean",
        )

    winner = max(heads, key=priority)
    final_cuts = reference_cuts("threshold", heads)
    snapshot(
        work,
        "frozen.json",
        {
            "protocol": inputs,
            "candidate": winner,
            "heads": heads,
            "thresholds": final_cuts,
            "select": selection,
            "pilot_replay": replay,
        },
    )


def encoder_id(path):
    files = sorted(
        p
        for p in path.iterdir()
        if p.is_file()
        and (p.suffix in {".safetensors", ".bin"} or p.name == "config.json")
    )
    if len(files) < 2:
        raise ValueError("Missing model config or weights")
    return {p.name: digest(p) for p in files}


def evaluate(root, work):
    from .adapters import selected_members
    from .audio import decode
    from .clip_cache import treat
    from .stage0 import groups
    from .teacher_eval import panel_rows

    frozen = json.loads((work / "frozen.json").read_text())
    v1 = root / V1
    original = reference(v1)
    binding = {
        "frozen": digest(work / "frozen.json"),
        "teacher_scores": digest(v1 / "runs/teacher-eval-clips.csv"),
        "teacher_freeze": digest(v1 / "runs/teacher-frozen.json"),
    }
    snapshot(work, "comparison-inputs.json", binding)
    ref = {(r["id"], r["channel"]): r for r in original}
    needed = {uid for uid, _ in ref}
    rows = [r for r in panel_rows(root, ["mlaad"]) if r["id"] in needed]
    if {r["id"] for r in rows} != needed or len(rows) != len(needed):
        raise ValueError("Reference panel IDs no longer resolve")
    path = root / "models/wavlm-base"
    if encoder_id(path) != frozen["protocol"]["encoder"]:
        raise ValueError("Encoder changed")
    checkpoint = work / "scores"
    checkpoint.mkdir(exist_ok=True)

    def output(uid):
        return checkpoint / (hashlib.sha256(uid.encode()).hexdigest() + ".json")

    pending = [r for r in rows if not output(r["id"]).exists()]
    if pending:
        net = Encoder(path)
        for i, (r, raw) in enumerate(
            selected_members(groups(root / "development", pending)), 1
        ):
            b = treat(raw)
            if b is None:
                raise ValueError("Previously eligible comparison clip failed")
            x = decode(b)
            scores = {}
            for view in VIEWS:
                if (r["id"], view) in ref:
                    features = net.feature(x, view)
                    scores[view] = {
                        name: predict(head, features)
                        for name, head in frozen["heads"].items()
                    }
            save(output(r["id"]), scores)
            if i % 100 == 0:
                print(f"Outside panel {i}/{len(pending)}", flush=True)
    scored = defaultdict(list)
    for row in rows:
        values = json.loads(output(row["id"]).read_text())
        for v, predictions in values.items():
            base = ref[row["id"], v]
            if base["label"] != row["label"] or set(predictions) != set(
                frozen["heads"]
            ):
                raise ValueError("Invalid comparison score")
            for name, s in predictions.items():
                if not np.isfinite(s):
                    raise ValueError("Nonfinite score")
                scored[name].append(dict(base, p=s))
    if any(len(scored[name]) != len(ref) for name in frozen["heads"]):
        raise ValueError("Incomplete paired scores")
    hi = json.loads((v1 / "runs/teacher-frozen.json").read_text())["thresholds"][1]
    report = {
        "candidate": frozen["candidate"],
        "probe": table(
            scored[frozen["candidate"]], frozen["thresholds"][frozen["candidate"]]
        ),
        "original_pilot": table(scored["clean"], frozen["thresholds"]["clean"]),
        "teacher_v1": table(original, hi),
        "inputs": binding,
        "scope": "Previously inspected English MLAAD development panel, including simulated channels. Real calls and independent confirmation unmeasured.",
    }
    save(work / "result.json", report)
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["baseline", "fit", "evaluate"])
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    work = a.data / "v2/day-target-1"
    work.mkdir(exist_ok=True, parents=True)
    if a.phase == "baseline":
        v1 = a.data / V1
        cut = json.loads((v1 / "runs/teacher-frozen.json").read_text())["thresholds"][1]
        save(work / "baseline.json", table(reference(v1), cut))
    elif a.phase == "fit":
        fit(a.data, work)
    else:
        evaluate(a.data, work)


if __name__ == "__main__":
    main()
