from pathlib import Path

import numpy as np
import pytest

from pipeline_v3 import phonemedf as ph
from pipeline_v3.audio import SR, WINDOW, fixed_crop, model_view
from pipeline_v3.match import assign
from pipeline_v3.voice_labels import classify


def example(reader, group):
    uid = reader + "-1234-0000"
    return {
        "uid": uid,
        "group": group,
        "reader": reader,
        "chapter": "1234",
        "member": group + "/" + uid + ".wav",
        "archive": ph.archive_name(group),
        "transcript": "sentence " + reader if group == "real" else "",
    }


def test_voice_and_family_allocation():
    rows = []
    labels = {}
    for fold in ph.FOLDS:
        reader = next(
            str(i) for i in range(1, 5000) if assign("librispeech/" + str(i)) == fold
        )
        for group in ["real", *ph.SYSTEMS]:
            r = example(reader, group)
            rows.append(r)
            labels[group, r["uid"]] = next(
                v for v, s in ph.VOICE_FOLD.items() if s == fold
            )
    pilot = ph.match(rows, labels, [])
    assert len(pilot) == 19
    assert not any(
        r["generator"] == "chatterboxtts" and r["split"] == "train" for r in pilot
    )
    assert len(ph.match(rows, labels, [], True)) == 20
    for k in labels:
        labels[k] = "unknown"
    assert not ph.match(rows, labels, [])
    for k in labels:
        labels[k] = "builtin-cohort"
    assert {r["split"] for r in ph.match(rows, labels, [])} == {"train"}


def test_open_set_requires_agreement_margin_and_floor():
    anchors = np.eye(10)
    assert classify(anchors[[0, 0]], anchors)[0] == ph.REFS[0]
    assert classify(anchors[[0, 1]], anchors)[0] == "unknown"
    assert classify(anchors[[0, 0]] * 0.84, anchors)[0] == "unknown"
    v = anchors[[0, 0]].copy()
    v[:, 1] = 0.95
    assert classify(v, anchors)[0] == "unknown"


def test_controls_equal_unpadded_length(monkeypatch):
    import io
    import wave

    for seconds in (2, 6, 18):
        # Exercise numerical preprocessing without generating a recording on disk.
        x = np.sin(np.arange(seconds * SR, dtype="float32") / 10)
        monkeypatch.setattr("pipeline_v3.audio.decode", lambda _, x=x: x)
        b = fixed_crop(b"fixture")
        if seconds == 2:
            assert b is None
        else:
            with wave.open(io.BytesIO(b), "rb") as w:
                assert w.getnframes() == WINDOW
            assert len(model_view(x, {"pair": "phonemedf"})) == WINDOW


def test_per_pair_gate_catches_hidden_shortcut():
    from pipeline_v3.audit import panels

    rows = []
    features = []
    for lang in ("en", "hi"):
        for pair in ("a", "b"):
            for split in ("train", "select"):
                for label in ("0", "1"):
                    rows.append(
                        {
                            "id": str(len(rows)),
                            "label": label,
                            "parent": lang + pair + split,
                            "split": split,
                            "language": lang,
                            "channel": "clean",
                            "pair": pair,
                        }
                    )
                    features.append(
                        {
                            "source": "common",
                            "pair": pair,
                            "duration": int(label) if pair == "a" else 1 - int(label),
                        }
                    )
    result = panels(rows, features)
    assert not result["pass"]
    assert result["source_pairs"]["a"]["controls"]["duration"]["auc"] == 1


def test_transcript_and_disjoint_inventory(tmp_path):
    for group in ["real", *ph.SYSTEMS, "dev-clean", "dev-other", "references"]:
        names = ["19-2-0003.wav"]
        if group.startswith("dev"):
            names = [("20" if group == "dev-clean" else "21") + "-3-0004.flac"]
        if group == "references":
            names = [v + "_023.wav" for v in ph.REFS]
        listing = tmp_path / (
            Path(ph.archive_name(group)).name.removesuffix(".tar.gz") + ".listing.txt"
        )
        listing.write_text(
            "".join(f"-rw-r--r-- 0/0 500 2000-01-01 12:00 {group}/{n}\n" for n in names)
        )
    (tmp_path / "19-2-0003.txt").write_text("This is the source text.")
    result = ph.inventory(None, tmp_path, tmp_path)
    assert (
        next(r for r in result if r["group"] == "real")["transcript"]
        == "This is the source text."
    )
    path = tmp_path / "dev-other.listing.txt"
    path.write_text(path.read_text().replace("21-3-0004", "19-3-0004"))
    with pytest.raises(ValueError, match="overlap"):
        ph.inventory(None, tmp_path)


def test_stage3_jobs_reserve_shared_text_and_gemini_train(tmp_path):
    from pipeline_v3.stage0_jobs import addon_jobs

    (tmp_path / "inventory.jsonl").touch()
    rows = []
    for i in range(100):
        r = example(str(i), "dev-clean")
        r["transcript"] = f"one distinct line read for this pilot by reader {i}"
        rows.append(r)
    rows.append(dict(rows[0], group="real"))
    early = addon_jobs(tmp_path, [], "edge", False, rows)
    final = addon_jobs(tmp_path, [], "edge", True, rows)
    assert early and all(j["text"] != rows[0]["transcript"] for j in early)
    assert not any(j["engine"] == "gemini" and j["split"] == "train" for j in early)
    assert any(j["engine"] == "gemini" and j["split"] == "train" for j in final)
    assert all(j["model"] == "edge-tts" for j in early if j["engine"] == "edge")
    assert {j["id"] for j in early if j["split"] != "train"} == {
        j["id"] for j in final if j["split"] != "train"
    }
