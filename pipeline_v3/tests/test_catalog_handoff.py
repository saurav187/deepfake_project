import json
import sys

import pytest

from pipeline_v3 import stage0
from pipeline_v3.cli import bound_audit
from pipeline_v3.data import digest, entries, payload, stage
from pipeline_v3.pilot import pilot_decisions
from pipeline_v3.tests.test_pipeline import record


def test_pilot_scores_each_candidate_for_original_languages():
    def result(language_eer):
        return {
            "selected": 10,
            "eligible": 10,
            "slices": {
                "domain/unseen": {"eer": 0.15},
                "language/en/unseen": {"eer": language_eer},
            },
        }

    models = {"xlsr/lfcc/linear/clean": {"slices": {"domain/unseen": {"eer": 0.20}}}}
    for tag, eer in (("a", 0.26), ("b", 0.25)):
        for view in ("clean", "opus"):
            models[f"{tag}/full/linear/{view}"] = result(eer)
    rows = [{"split": "train", "label": label} for label in ("0", "1")] * 1000
    assert pilot_decisions(models, rows, ["en"]) == {
        "a/full/linear/clean": False,
        "b/full/linear/clean": True,
    }


def test_voice_labels_reject_changed_shared_inventory(tmp_path):
    shared = tmp_path / "shared.csv"
    shared.write_text("old")
    (tmp_path / "inventory.csv").write_text("old")
    (tmp_path / "voice-inventory.sha256").write_text(digest(shared))
    stage0.write_csv(
        tmp_path / "voices.csv", [{"group": "XTTSv2", "uid": "a", "voice": "p236"}]
    )
    rows = [{"group": "XTTSv2", "uid": "a"}]
    assert stage0.load_labels(tmp_path, rows, shared)
    shared.write_text("changed metadata, same ids")
    with pytest.raises(ValueError, match="inventory mismatch"):
        stage0.load_labels(tmp_path, rows, shared)


def test_xtts_pack_recovery_and_live_binding(tmp_path, monkeypatch):
    work = tmp_path / "stage0-en-ecapa-xtts"
    root = tmp_path / "development"
    work.mkdir()
    root.mkdir()
    rows = []
    for fold in ("train", "select", "calibrate", "threshold", "validate"):
        for i in range(20 if fold == "train" else 5):
            parent = f"{fold}-{i}"
            for label in (0, 1):
                rid = f"{parent}-{label}"
                b = rid.encode()
                (root / rid).write_bytes(b)
                row = dict(
                    record(label, fold),
                    id=rid,
                    member="-",
                    archive=rid,
                    path=rid,
                    speaker=parent,
                    parent=parent,
                    text=parent,
                    pair="phonemedf",
                    language="en",
                    generator="-"
                    if not label
                    else ("xtts_v2" if fold == "train" else "chatterboxtts"),
                    domain="real"
                    if not label
                    else ("seen" if fold == "train" else "unseen"),
                )
                rows.append(row)
    stage0.write_csv(work / "selection.csv", rows)
    monkeypatch.setattr(stage0, "fixed_crop", lambda b: b + b"-crop")
    monkeypatch.setattr(stage0, "descriptors", lambda b: {"duration": 4})
    stage0.audit_selection(root, work, ["en"], stage0.XTTS_ONLY_TRAIN)
    stage0.catalog(root, work, ["en"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage0",
            "pack",
            "--data",
            str(tmp_path),
            "--languages",
            "en",
            "--voices",
            "ecapa",
            "--xtts-only",
        ],
    )
    stage0.main()
    pack = work / "packed"
    index = pack / "index.json"
    (work / "audit.json").unlink()
    stage0.main()
    assert bound_audit(work, index)["pass"]
    stage(pack, tmp_path / "local", {"train", "select"})
    for row in entries(tmp_path / "local", {"train", "select"}):
        assert payload(tmp_path / "local", row) == row["id"].encode() + b"-crop"

    # Re-running match or audit invalidates the old authorization.
    selection = work / "selection.csv"
    original = selection.read_bytes()
    selection.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="changed since pack"):
        bound_audit(work, index)
    selection.write_bytes(original)
    prep = work / "prep-audit.json"
    original = prep.read_bytes()
    prep.write_text(json.dumps(dict(json.loads(original), **{"pass": False})))
    with pytest.raises(ValueError, match="changed since pack"):
        bound_audit(work, index)
    prep.write_bytes(original)
    catalog = work / "catalog.csv"
    original = catalog.read_bytes()
    catalog.write_bytes(
        original.replace(b"stage0-en-ecapa-xtts/bytes/", b"stage0-en-ecapa/bytes/")
    )
    with pytest.raises(ValueError, match="different experiment"):
        stage0.main()
    catalog.write_bytes(original)
    stale = json.loads(index.read_text())
    stale[0]["parent"] = "other"
    index.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="Existing pack differs"):
        stage0.main()
    rows[1]["generator"] = "melotts"
    with pytest.raises(ValueError, match="Unexpected row"):
        stage0.viable(rows, stage0.XTTS_ONLY_TRAIN)
