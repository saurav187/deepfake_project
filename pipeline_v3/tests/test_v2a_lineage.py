from pipeline_v3.v2a_lineage import resolve


def test_resolve_joins_librisevoc_and_withholds_unverified_asv(tmp_path):
    d = tmp_path / "train-clean-100" / "103" / "1241"
    d.mkdir(parents=True)
    (d / "103_1241_000004_000002.normalized.txt").write_text("Hello world.")
    reqs = [
        dict(source="librisevoc", uid="103_1241_000004_000002", speaker="librispeech/103"),
        dict(source="librisevoc", uid="103_1241_000009_000000", speaker="librispeech/103"),
        dict(source="asvspoof5", uid="T_0000000001", speaker=""),
    ]
    out, missing = resolve(reqs, tmp_path, "md5x")
    assert [r["uid"] for r in out] == ["103_1241_000004_000002"]
    assert out[0]["transcript"] == "Hello world." and "md5x" in out[0]["evidence"]
    assert missing == {"librisevoc": 1, "asvspoof5": 1}


def test_resolve_rejects_path_key_mismatch(tmp_path):
    import pytest

    d = tmp_path / "train-clean-100" / "999" / "1241"
    d.mkdir(parents=True)
    (d / "103_1241_000004_000002.normalized.txt").write_text("x")
    with pytest.raises(ValueError):
        resolve([], tmp_path, "m")
