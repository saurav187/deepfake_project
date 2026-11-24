import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from .audio import decode, prepare
from .data import digest, entries, payload
from .metrics import bootstrap, metrics, slices
from .models import Student, Teacher
from .release import Detector
from .train import logits, score


def platt(x, y):
    m = LogisticRegression(C=100, class_weight="balanced")
    m.fit(np.asarray(x).reshape(-1, 1), y)
    if m.coef_[0, 0] <= 0:
        raise ValueError("Development logits do not support monotone calibration")
    return [float(m.coef_[0, 0]), float(m.intercept_[0])]


def export(cfg):
    source = cfg.data / "runs" / "student.best.pt"
    model = Student().eval()
    model.load_state_dict(torch.load(source, map_location="cpu", weights_only=True))
    # Linear layers (all learned parameters except biases) have packed int8 weights.
    model = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    out = cfg.release
    if out.exists():
        raise ValueError("Release destination already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        path = Path(temp) / "release"
        path.mkdir()
        artifact = path / "student.ts"
        torch.jit.script(model).save(str(artifact))
        deployed = torch.jit.load(str(artifact)).eval()

        def collect(split):
            rows = entries(cfg.local, {split})
            return score(deployed, True, rows, cfg.local, "cpu")

        cal = collect("calibrate")
        a, b = platt([s.mean() for _, s in cal], [int(r["label"]) for r, _ in cal])
        window = platt(
            [v for _, s in cal for v in s], [int(r["label"]) for r, s in cal for _ in s]
        )
        thresholds = collect("threshold")
        # Only long enough recordings receive determinate verdicts.
        groups = [
            [
                expit(a * s.mean() + b)
                for r, s in thresholds
                if r["label"] == lab and r["length"] == "4s+"
            ]
            for lab in ("0", "1")
        ]
        if min(map(len, groups)) < 50:
            raise ValueError(
                "Need at least 50 independent-fold recordings per class at threshold selection"
            )
        lo = min(0.40, float(np.quantile(groups[1], 0.02, method="lower")))
        hi = max(
            0.60,
            float(np.nextafter(np.quantile(groups[0], 0.98, method="higher"), 1.0)),
        )
        rule = {
            "file": [a, b],
            "window": window,
            "thresholds": [lo, hi],
            "sha256": digest(artifact),
            "training_index": digest(cfg.local / "index.json"),
            "source_sha256": digest(source),
        }
        (path / "release.json").write_text(json.dumps(rule, indent=2))
        if sum(f.stat().st_size for f in path.iterdir()) > 90_000_000:
            raise ValueError("Release exceeds 90 decimal MB")
        Detector(path)
        path.rename(out)
    print(out)


def evaluate(cfg):
    d = Detector(cfg.release)
    rows = entries(cfg.local, {"validate"})
    result = [d.analyze(payload(cfg.local, r), "") for r in rows]
    valid = [
        (dict(r, length="<4s" if s["speech_s"] < 4 else "4s+"), s)
        for r, s in zip(rows, result)
        if s["fake_probability"] is not None
    ]
    rr = [r for r, _ in valid]
    probs = [s["fake_probability"] for _, s in valid]
    y = [int(r["label"]) for r in rr]
    out = {
        "student": metrics(y, probs),
        "by": slices(rr, probs),
        "ci": bootstrap(rr, probs),
        "total": len(rows),
        "scored": len(rr),
        "brier": float(np.mean((np.array(y) - probs) ** 2)),
        "bytes": sum(f.stat().st_size for f in cfg.release.iterdir() if f.is_file()),
    }
    out["student"]["accuracy_at_reference_half"] = float(
        np.mean((np.array(probs) >= 0.5) == np.array(y))
    )
    out["covered"] = sum(r["verdict"] in {"likely_real", "likely_fake"} for r in result)
    out["validation_slices"] = {}
    for field in ("language", "channel", "pair"):
        for value in sorted({r[field] for r in rows}):
            cohort = [
                (int(r["label"]), s["verdict"])
                for r, s in zip(rows, result)
                if r[field] == value
            ]
            out["validation_slices"][field + "/" + value] = {
                "total": len(cohort),
                "genuine": sum(y == 0 for y, v in cohort),
                "false_accusations": sum(
                    y == 0 and v == "likely_fake" for y, v in cohort
                ),
                "false_reassurances": sum(
                    y == 1 and v == "likely_real" for y, v in cohort
                ),
                "determinate": sum(
                    v in {"likely_real", "likely_fake"} for y, v in cohort
                ),
            }
    out["reliability"] = []
    for i in range(10):
        idx = np.flatnonzero(np.minimum(9, (np.asarray(probs) * 10).astype(int)) == i)
        if len(idx):
            out["reliability"].append(
                {
                    "bin": i,
                    "n": len(idx),
                    "mean_probability": float(np.mean(np.asarray(probs)[idx])),
                    "fraction_fake": float(np.mean(np.asarray(y)[idx])),
                }
            )
    for lab in ("0", "1"):
        sub = [s for r, s in zip(rows, result) if r["label"] == lab]
        out["outcomes_" + lab] = {
            k: sum(s["verdict"] == k for s in sub)
            for k in ("likely_fake", "likely_real", "uncertain", "insufficient_audio")
        }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = Teacher(cfg.encoder).to(device)
    path = cfg.data / "runs" / "teacher.best.pt"
    teacher.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    reference = score(teacher, False, rows, cfg.local, device)
    out["teacher"] = metrics(
        [int(r["label"]) for r, _ in reference], [s.mean() for _, s in reference]
    )
    out["teacher"]["accuracy_at_zero_logit"] = float(
        np.mean([(s.mean() >= 0) == bool(int(r["label"])) for r, s in reference])
    )
    out["teacher"]["bytes"] = path.stat().st_size
    # Same synthetic 30-second input, CPU threads, preparation and window schedule.
    torch.set_num_threads(cfg.threads)
    teacher.cpu().eval()
    sample = benchmark_input()
    with torch.inference_mode():
        teacher_times = []
        for _ in range(4):
            start = time.perf_counter()
            x, _ = prepare(decode(sample))
            logits(teacher, x, False, "cpu")
            teacher_times.append(time.perf_counter() - start)
    out["teacher"]["cpu_warm_median_30s"] = float(np.median(teacher_times[1:]))
    out["student"]["cpu_warm_median_30s"] = benchmark(cfg.release, cfg.threads)[
        "seconds_30s_warm_median"
    ]
    (cfg.data / "validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def benchmark_input():
    import io
    import wave

    x = np.random.default_rng(123).normal(0, 2000, 30 * 16000).astype("<i2")
    stream = io.BytesIO()
    with wave.open(stream, "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", ""))
        f.writeframes(x.tobytes())
    return stream.getvalue()


def benchmark(release, threads):
    torch.set_num_threads(threads)
    d = Detector(release)
    b = benchmark_input()
    times = []
    for _ in range(7):
        t = time.perf_counter()
        d.analyze(b, "bench.wav")
        times.append(time.perf_counter() - t)
    report = {
        "seconds_30s_warm_median": float(np.median(times[1:])),
        "cold_seconds": times[0],
        "threads": threads,
    }
    print(json.dumps(report))
    return report
