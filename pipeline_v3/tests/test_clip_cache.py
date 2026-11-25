import json
import zipfile

import numpy as np
import pytest

from pipeline_v3 import clip_cache as cc
from pipeline_v3.audio import SR
from pipeline_v3.en_treatment import _wav


def make_zip(tmp_path):
    z = tmp_path / "src.zip"
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(z, "w") as f:
        for i in range(5):
            f.writestr(f"gt/a{i}.wav", _wav(0.1 * rng.standard_normal(6 * SR)))
        f.writestr("gt/short.wav", _wav(0.1 * rng.standard_normal(SR)))
    return z


def test_cache_publishes_records_and_resume_skips(tmp_path):
    z = make_zip(tmp_path)
    c = cc.ClipCache(tmp_path / "data")
    items = [(f"gt/a{i}.wav", f"gt/a{i}.wav") for i in range(5)] + [("gt/short.wav", "gt/short.wav")]
    st = c.ensure_zip("librisevoc", z, items[:3], tmp_path, workers=2, shard_size=2, log=lambda m: None)
    assert st["processed"] == 3
    st2 = c.ensure_zip("librisevoc", z, items, tmp_path, workers=2, shard_size=10, log=lambda m: None)
    assert st2["skipped_cached"] == 3 and st2["processed"] == 3
    again = cc.ClipCache(tmp_path / "data")  # reload from Drive manifests
    assert again.ensure_zip("librisevoc", z, items, tmp_path, workers=2, log=lambda m: None)["processed"] == 0
    with zipfile.ZipFile(z) as f:
        assert again.get("librisevoc", "gt/a1.wav") == cc.treat(f.read("gt/a1.wav"))
    e = again.entries["librisevoc", "gt/short.wav"]
    assert not e["eligible"] and e["reason"] == "shorter_than_4s_after_trim"
    assert again.get("librisevoc", "gt/short.wav") is None
    assert all(r["prep_version"] == cc.PREP_VERSION for r in again.entries.values())


def test_cache_detects_corruption_and_code_change(tmp_path):
    z = make_zip(tmp_path)
    c = cc.ClipCache(tmp_path / "data")
    c.ensure_zip("librisevoc", z, [("gt/a0.wav", "gt/a0.wav")], tmp_path, workers=1, log=lambda m: None)
    shard = next((c.root / "shards").glob("*.tar"))
    raw = bytearray(shard.read_bytes())
    e = c.entries["librisevoc", "gt/a0.wav"]
    raw[e["offset"] + 100] ^= 1
    shard.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        cc.ClipCache(tmp_path / "data").get("librisevoc", "gt/a0.wav")
    meta = c.root / "meta.json"
    meta.write_text(json.dumps(dict(json.loads(meta.read_text()), fingerprint="old")))
    with pytest.raises(ValueError):
        cc.ClipCache(tmp_path / "data")
