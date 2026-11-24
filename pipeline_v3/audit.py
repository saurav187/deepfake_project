"""Apply nuisance-only controls fitted on train and scored on separate selection speakers."""

import json
import subprocess

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .audio import SR, decode
from .data import entries, payload
from .metrics import metrics


def descriptors(b):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-protocol_whitelist",
        "pipe",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate",
        "-of",
        "json",
        "pipe:0",
    ]
    meta = json.loads(
        subprocess.run(cmd, input=b, capture_output=True, check=True, timeout=30).stdout
    )["streams"][0]
    x = decode(b)
    rms = float(np.sqrt(np.mean(x**2)))
    active = np.flatnonzero(np.abs(x) > max(0.001, 0.03 * np.max(abs(x))))
    fft = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x), 1 / SR)
    energy = fft.sum() + 1e-10
    return {
        "codec": meta["codec_name"],
        "rate": str(meta["sample_rate"]),
        "duration": len(x) / SR,
        "rms": rms,
        "edge": 1.0
        if not len(active)
        else (active[0] + len(x) - 1 - active[-1]) / len(x),
        "upper_band": fft[freq >= 4000].sum() / energy,
        "speech_ratio": float(
            np.mean(
                np.sqrt(
                    np.mean(
                        np.pad(x, (0, (-len(x)) % 160)).reshape(-1, 160) ** 2, axis=1
                    )
                )
                > max(0.001, 0.03 * np.max(abs(x)))
            )
        ),
        "rolloff_hz": float(
            freq[min(len(freq) - 1, np.searchsorted(np.cumsum(fft), 0.95 * fft.sum()))]
        ),
        "tilt": fft[freq <= 1000].sum() / energy,
    }


def audit(local):
    rows = entries(local, {"train", "select"})
    features = [
        dict(descriptors(payload(local, r)), pair=r["pair"], source=r["source"])
        for r in rows
    ]
    return panels(rows, features)


def panels(rows, features):
    result = fit_controls(rows, features)
    result["languages"] = {}
    for lang in ("en", "hi"):
        idx = [i for i, r in enumerate(rows) if r["language"] == lang]
        panel = [rows[i] for i in idx]
        if {(r["split"], r["label"]) for r in panel} != {
            (s, y) for s in ("train", "select") for y in ("0", "1")
        }:
            raise ValueError(f"Missing real/fake audit cell for {lang}")
        result["languages"][lang] = fit_controls(panel, [features[i] for i in idx])
    result["pass"] &= all(item["pass"] for item in result["languages"].values())
    result["source_pairs"] = {}
    for pair in sorted({r["pair"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["pair"] == pair]
        subset = [rows[i] for i in idx]
        if {(r["split"], r["label"]) for r in subset} != {
            (s, y) for s in ("train", "select") for y in ("0", "1")
        }:
            raise ValueError(f"Incomplete audit panel: {pair}")
        result["source_pairs"][pair] = fit_controls(subset, [features[i] for i in idx])
    result["pass"] &= all(v["pass"] for v in result["source_pairs"].values())
    return result


def fit_controls(rows, features):
    train = np.array([r["split"] == "train" for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    reports = {}
    for field in [*features[0], "joint"]:
        # Stage 0 gates packaging provenance as well as acoustic measurements.
        inputs = [
            {k: v for k, v in f.items() if k == field or field == "joint"}
            for f in features
        ]
        model = make_pipeline(
            DictVectorizer(sparse=False),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=500),
        )
        model.fit([x for x, t in zip(inputs, train) if t], y[train])
        scores = model.decision_function([x for x, t in zip(inputs, train) if not t])
        reports[field] = metrics(y[~train], scores)
    return assess(rows, reports)


def assess(rows, reports):
    # Both orientations are shortcuts. Missing measurements fail closed.
    def informative(auc):
        return auc is None or max(auc, 1 - auc) > 0.60

    real = {
        (r["split"], r["pair"], r["language"], r["channel"], parent)
        for r in rows
        if r["label"] == "0"
        for parent in r["parent"].split("|")
    }
    unmatched = [
        r["id"]
        for r in rows
        if r["label"] == "1"
        and not any(
            (r["split"], r["pair"], r["language"], r["channel"], p) in real
            for p in r["parent"].split("|")
        )
    ]
    source_confound = informative(reports["source"]["auc"])
    return {
        "controls": reports,
        "unmatched_fake_parents": len(unmatched),
        "provenance_warning": source_confound,
        "pass": all(not informative(r["auc"]) for r in reports.values())
        and not unmatched,
    }
