"""Regression tests for V2-90 controls (synthetic shards; no GPU, no real audio)."""

import json

import numpy as np
import pytest

from pipeline_v3 import v2_90 as v


def _rows(fams, n, label=1, split="dev", source="mlaad"):
    out = []
    for f in fams:
        for i in range(n):
            out.append({"uid": f"{f}/{i}", "family": f, "label": label, "split": split, "source": source,
                        "speaker": f"{f}:v{i % 3}", "parent": f"{f}/p{i}"})
    return out


def test_dev_metric_one_family_is_insufficient():
    rows = _rows(["Chatterbox"], 40) + _rows(["genuine"], 40, label=0, source="vctk")
    sc = [1.0] * 40 + [0.0] * 40
    m = v.dev_metric(rows, sc)
    assert not m["sufficient"] and m["key"] == -1.0
    assert any("ElevenLabs-v3" in s for s in m["insufficient"])


def test_dev_metric_all_families_sufficient():
    rows = _rows(v.TARGETS, 30) + _rows(["genuine"], 40, label=0, source="vctk")
    m = v.dev_metric(rows, [1.0] * 120 + [0.0] * 40)
    assert m["sufficient"] and m["key"] == 1.0


def _write_shards(d, n=3):
    sh = d / "shards"
    sh.mkdir(parents=True)
    arr = np.zeros((n, v.WIN), dtype="<i2")
    np.save(sh / "test-0000.npy", arr)
    rows = [{"uid": f"u{i}", "split": "test", "label": 0, "family": "genuine", "source": "vctk",
             "speaker": f"s{i}", "parent": f"p{i}", "shard": "test-0000.npy", "idx": i} for i in range(n)]
    (sh / "test-0000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "prep.json").write_text("{}")
    return sh


def test_prepared_manifest_detects_tampering(tmp_path):
    work = tmp_path / "w"
    stage = work / "stage-x"
    _write_shards(stage)
    v.publish_prepared(work, stage)
    fp = v.verify_prepared(work)
    assert fp
    npy = work / "prepared/shards/test-0000.npy"
    a = np.load(npy)
    a[0, 0] = 7
    np.save(npy, a)
    with pytest.raises(ValueError):
        v.verify_prepared(work)


def test_unpublished_prep_is_rejected(tmp_path):
    work = tmp_path / "w"
    _write_shards(work / "prepared")  # shards present but no PREPARED.json (interrupted copy)
    with pytest.raises(ValueError, match="completion manifest"):
        v.verify_prepared(work)


def test_changed_shard_index_is_rejected(tmp_path):
    sh = _write_shards(tmp_path / "s")
    rows = [json.loads(line) for line in (sh / "test-0000.jsonl").read_text().splitlines()]
    rows[1]["idx"] = 0
    (sh / "test-0000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        v.verify_shards(sh)


def test_duplicate_recording_and_split_crossing_rejected():
    base = {"label": 0, "family": "genuine", "source": "vctk"}
    dup = [dict(base, uid="a", split="train", speaker="s1", parent="p1"),
           dict(base, uid="a", split="test", speaker="s2", parent="p2")]
    with pytest.raises(ValueError, match="duplicate"):
        v.leakage(dup)
    cross = [dict(base, uid="a", split="train", speaker="s1", parent="p1", group="g1"),
             dict(base, uid="b", split="test", speaker="s2", parent="p2", group="g1")]
    with pytest.raises(ValueError, match="leakage"):
        v.leakage(cross)


def _signed_work(tmp_path):
    """Published prepared set with a test split, plus a signoff written by the real signoff()."""
    import hashlib

    work = tmp_path / "w"
    stage = work / "stage-x"
    _write_shards(stage, n=4)
    v.publish_prepared(work, stage)
    (work / "ckpt-c2").mkdir()
    (work / "ckpt-c2/best.pt").write_bytes(b"weights")
    sel = {"members": [{"tag": "c2", "sha256": hashlib.sha256(b"weights").hexdigest()}]}
    (work / "selection.json").write_text(json.dumps(sel))
    (work / "cutoff.json").write_text(json.dumps({"model_sha256": v.fsha(work / "selection.json"), "cutoff": 0.5}))
    proto = tmp_path / "p.md"
    proto.write_text("protocol")
    v.signoff(work, proto, "test")
    return work


def test_signoff_rejects_changed_shard_bytes(tmp_path):
    work = _signed_work(tmp_path)
    v.check_signoff(work)
    npy = work / "prepared/shards/test-0000.npy"
    a = np.load(npy)
    a[1, 5] = 3
    np.save(npy, a)
    with pytest.raises(ValueError):
        v.check_signoff(work)


def test_signoff_binding_changes_with_shard_index(tmp_path):
    work = _signed_work(tmp_path)
    tb0, _, _ = v.test_binding(work)
    j = work / "prepared/shards/test-0000.jsonl"
    rr = [json.loads(line) for line in j.read_text().splitlines()]
    rr[0]["idx"], rr[1]["idx"] = 1, 0  # rows now point at each other's audio
    j.write_text("".join(json.dumps(r) + "\n" for r in rr))
    tb1, _, _ = v.test_binding(work)
    assert tb1 != tb0
    with pytest.raises(ValueError):
        v.check_signoff(work)


def test_duplicate_test_recording_rejected(tmp_path):
    work = _signed_work(tmp_path)
    j = work / "prepared/shards/test-0000.jsonl"
    rr = [json.loads(line) for line in j.read_text().splitlines()]
    rr[1]["uid"] = rr[0]["uid"]
    j.write_text("".join(json.dumps(r) + "\n" for r in rr))
    with pytest.raises(ValueError, match="duplicate"):
        v.test_binding(work)


def test_repeated_evaluation_is_preserved(tmp_path):
    work = _signed_work(tmp_path)
    calls = []

    def scorer(st, rows, ch):
        calls.append(ch)
        return [0.0 for _ in rows]

    r1 = v.evaluate(work, scorer=scorer, channels=["clean"])
    r2 = v.evaluate(work, scorer=scorer, channels=["clean"])
    assert calls == ["clean"] and r1 == r2
    assert "INSUFFICIENT" in r1["channels"]["clean"]["verdict"]  # 4 genuine rows, no target family
