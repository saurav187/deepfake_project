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

# Audit policy (owner decision, 25 Nov 2026, "Option A"). Literal provenance fields
# (which corpus, which matched source pair, archive or path) are reported as a
# prominent provenance warning but no longer block training: in matched public
# corpora such as Kathbath/IndicSynth, real and fake necessarily come from different
# releases, so source-only AUC is 1.00 by construction and is not something the
# model can observe. Every measurement of the decoded signal, plus the stored codec
# and sample rate, still blocks at the 0.60 cutoff in both directions, alone and in
# a joint predictor that excludes provenance fields. Older reports
# ("source-strict-v1") cannot authorize pilot or training under this version.
POLICY = "acoustic-gate-v2-provenance-warning"
PROVENANCE_FIELDS = frozenset({"source", "pair", "archive", "path"})


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


def panels(rows, features, languages=("en", "hi")):
    """Audit every configured language; rows from any other language fail closed."""
    languages = tuple(languages)
    if not languages or {r["language"] for r in rows} - set(languages):
        raise ValueError("Audit rows contain an unconfigured language")
    result = fit_controls(rows, features)
    result["languages"] = {}
    result["languages_audited"] = sorted(languages)
    for lang in languages:
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
    result["policy"] = POLICY
    result["provenance_warning"] = result.get("provenance_warning") or any(
        v["provenance_warning"]
        for v in [*result["languages"].values(), *result["source_pairs"].values()]
    )
    return result


def fit_controls(rows, features):
    train = np.array([r["split"] == "train" for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    reports = {}
    for field in [*features[0], "joint"]:
        # The joint predictor combines every blocking field and excludes provenance.
        inputs = [
            {
                k: v
                for k, v in f.items()
                if k == field or (field == "joint" and k not in PROVENANCE_FIELDS)
            }
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
    provenance = {k: v for k, v in reports.items() if k in PROVENANCE_FIELDS}
    blocking = {k: v for k, v in reports.items() if k not in PROVENANCE_FIELDS}
    failed = sorted(k for k, v in blocking.items() if informative(v["auc"]))
    return {
        "policy": POLICY,
        "controls": reports,
        "blocking_fields": sorted(blocking),
        "failed_fields": failed,
        "unmatched_fake_parents": len(unmatched),
        # Reported prominently, never silently: literal provenance predicts the label.
        "provenance_warning": any(informative(v["auc"]) for v in provenance.values()),
        "provenance_auc": {k: v["auc"] for k, v in provenance.items()},
        "pass": not failed and not unmatched,
    }


def require_passing(report, index_sha256):
    """Gate for pilot/teacher/student: a passing audit under this policy, for this pack."""
    if (
        report.get("policy") != POLICY
        or not report.get("pass")
        or report.get("index_sha256") != index_sha256
    ):
        raise ValueError(
            f"A passing {POLICY} audit for this pack is required "
            f"(found policy {report.get('policy')!r}, pass={report.get('pass')})"
        )
