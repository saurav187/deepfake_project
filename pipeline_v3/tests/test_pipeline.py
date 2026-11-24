import csv
import io
import json
import wave

import numpy as np
import pytest
import torch

from pipeline_v3.audio import SR, prepare, windows
from pipeline_v3.data import (
    FIELDS,
    check,
    digest,
    entries,
    pack,
    payload,
    safe_path,
    stage,
)
from pipeline_v3.metrics import metrics
from pipeline_v3.models import Student


def record(label, split="train"):
    key = split + str(label)
    return {
        "id": key,
        "path": key,
        "split": split,
        "label": str(label),
        "source": "fixture",
        "pair": "book",
        "language": "hi",
        "channel": "clean",
        "speaker": key,
        "voice": key if label else "-",
        "parent": key,
        "text": key,
        "generator": "engine" if label else "-",
        "domain": "seen",
    }


@pytest.mark.parametrize("field", ["speaker", "voice", "parent", "text", "sha256"])
def test_leakage(field):
    rows = [record(l, s) for s in ["train", "select"] for l in (0, 1)]
    rows[1][field] = rows[3][field] = "collision"
    with pytest.raises(ValueError, match="leaks"):
        check(rows)


def test_holdout_and_matched_controls():
    rows = [record(l, s) for s in ["train", "select"] for l in (0, 1)]
    rows[-1]["domain"] = "unseen"
    with pytest.raises(ValueError, match="unseen"):
        check(rows)
    with pytest.raises(ValueError, match="Unmatched"):
        check(rows[:1])


def test_pack_stage_and_corruption(tmp_path):
    root = tmp_path / "development"
    root.mkdir()
    rows = [record(i) for i in (0, 1)]
    for r in rows:
        (root / r["path"]).write_bytes(r["id"].encode())
    with (root / "catalog.csv").open("w") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    packed = tmp_path / "packed"
    pack(tmp_path, "catalog.csv", packed, 10)
    local = tmp_path / "local"
    stage(packed, local, {"train"})
    assert [payload(local, r).decode() for r in entries(local, {"train"})] == [
        r["id"] for r in rows
    ]
    next(local.glob("*.tar")).write_bytes(b"")
    stage(packed, local, {"train"})
    assert all(payload(local, r) for r in entries(local, {"train"}))
    with pytest.raises(ValueError):
        safe_path(root, "../sealed/test.csv")


def test_window_offsets():
    x = np.concatenate([np.zeros(SR), np.ones(5 * SR), np.zeros(SR)]).astype("float32")
    y, start = prepare(x)
    clips = list(windows(y))
    assert start > SR and clips[-1][0] + len(clips[-1][1]) == len(y)
    assert metrics([0, 1], [0, 1])["eer"] == 0


def test_quantized_runtime(tmp_path, monkeypatch):
    model = torch.ao.quantization.quantize_dynamic(
        Student().eval(), {torch.nn.Linear}, dtype=torch.qint8
    )
    path = tmp_path / "student.ts"
    torch.jit.script(model).save(str(path))
    (tmp_path / "release.json").write_text(
        json.dumps(
            {
                "sha256": digest(path),
                "window": [1, 0],
                "file": [1, 0],
                "thresholds": [0.4, 0.6],
            }
        )
    )
    from app.backend.config import Settings
    from app.backend.detectors.registry import load_backend

    monkeypatch.setenv("V3_RELEASE", str(tmp_path))
    d = load_backend(Settings(backend="v3"))
    for seconds, expected in [(0.4, "insufficient_audio"), (2, "uncertain"), (5, None)]:
        f = io.BytesIO()
        with wave.open(f, "wb") as w:
            w.setparams((1, 2, SR, 0, "NONE", ""))
            w.writeframes(
                (np.sin(np.arange(int(seconds * SR)) / 16) * 8000)
                .astype("<i2")
                .tobytes()
            )
        result = d.analyze(f.getvalue(), "audio.wav")
        assert result["verdict"] == expected if expected else result["segments"]
        assert result["duration_s"] == seconds
    with pytest.raises(ValueError):
        d.analyze(b"invalid", "clip.mp4")


def test_shortcut_controls():
    from pipeline_v3.audit import assess

    rows = [record(i) for i in (0, 1)]
    controls = {"source": {"auc": 1.0}, "edge": {"auc": 0.5}}
    assert not assess(rows, controls)["pass"]
    rows[1]["parent"] = rows[0]["parent"]
    assert assess(rows, controls)["pass"]
    for value in (0.20, 0.85, None):
        assert not assess(rows, dict(controls, rate={"auc": value}))["pass"]


def test_match_join(tmp_path):
    from pipeline_v3.match import assign, match

    root = tmp_path / "development"
    root.mkdir()
    speaker = next(
        str(i) for i in range(1000) if assign("kathbath/" + str(i)) == "train"
    )
    name = f"123-{speaker}-m.wav"
    rows = [
        {
            "source": "kathbath",
            "archive": "kathbath/train.parquet",
            "member": "0",
            "metadata": {"audio_path": name, "text": "a line"},
        },
        {
            "source": "indicsynth",
            "archive": "indicsynth/train.parquet",
            "member": "3",
            "metadata": {
                "Generative Model": "xtts_v2",
                "Source Reference Audio": name,
                "Target Speaker ID": int(speaker),
                "TTS Transcript": "a line",
            },
        },
    ]
    (root / "inventory").write_text("\n".join(json.dumps(r) for r in rows))
    match(tmp_path, "inventory", "selection.csv", ["freevc24"])
    with (root / "selection.csv").open() as f:
        a, b = list(csv.DictReader(f))
    assert (
        a["parent"] == b["parent"] and a["text"] == b["text"] and b["domain"] == "seen"
    )


def test_video_container_in_memory(tmp_path):
    import subprocess

    from pipeline_v3.audio import decode
    from pipeline_v3.media import probe_duration

    path = tmp_path / "video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=600:duration=3",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )
    data = path.read_bytes()
    assert probe_duration(data, "upload.mp4") == pytest.approx(3, abs=0.05)
    assert len(decode(data)) / SR == pytest.approx(3, abs=0.05)
