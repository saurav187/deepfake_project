import hashlib
import json

import numpy as np
import pytest

from pipeline_v3 import en_treatment as et
from pipeline_v3.audio import SR, WINDOW, decode


def crop(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(WINDOW) / SR
    x = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(WINDOW)
    return et._wav(x)


@pytest.mark.parametrize("name", et.TREATMENTS)
def test_treatments_keep_the_exact_window_and_level(name):
    out = et.apply(crop(), name)
    x = decode(out)
    assert len(x) == WINDOW
    assert abs(float(np.sqrt(np.mean(x**2))) - et.LEVEL_RMS) < 0.02
    assert et.apply(crop(), name) == out  # deterministic, so catalog reproduces the audit


def test_lowpass_removes_energy_above_7k():
    rng = np.random.default_rng(1)
    noisy = et._wav(0.1 * rng.standard_normal(WINDOW))
    x = decode(et.apply(noisy, "norm_lp7k"))
    f = np.fft.rfftfreq(len(x), 1 / SR)
    p = np.abs(np.fft.rfft(x)) ** 2
    assert p[f > 7500].sum() / p.sum() < 0.01


def test_apply_rejects_wrong_length_and_unknown():
    with pytest.raises(ValueError):
        et.apply(et._wav(np.zeros(WINDOW - 160)), "norm")
    with pytest.raises(ValueError):
        et.apply(crop(), "aac")


def rows(readers=10, per=3):
    out = []
    for s in range(readers):
        for u in range(per):
            p = f"p{s}_{u}"
            out.append(dict(id=f"r{p}", parent=p, speaker=f"s{s}", label="0", split="train"))
            # A fake's speaker field may name a target voice; grouping uses the parent's reader.
            out.append(dict(id=f"f{p}", parent=p, speaker="vctk", label="1", split="train"))
    return out


def test_folds_are_reader_disjoint_and_keep_parents_together():
    rs = rows()
    fold = et.folds(rs)
    by_reader, by_parent = {}, {}
    for r in rs:
        by_parent.setdefault(r["parent"], set()).add(fold[r["id"]])
        if r["label"] == "0":
            by_reader.setdefault(r["speaker"], set()).add(fold[r["id"]])
    assert all(len(v) == 1 for v in by_parent.values())
    assert all(len(v) == 1 for v in by_reader.values())
    assert set(fold.values()) == set(range(et.FOLDS))
    assert et.folds(list(reversed(rs))) == fold


def test_fold_requires_genuine_parent():
    with pytest.raises(ValueError):
        et.folds([dict(id="f", parent="x", speaker="v", label="1", split="train")])


def test_score_uses_both_directions_and_fails_closed():
    total, mean = et.score([{"rms": 0.2, "joint": 0.55}, {"rms": 0.3, "joint": None}])
    assert mean["rms"] == pytest.approx(0.75)
    assert mean["joint"] == pytest.approx((0.55 + 1.0) / 2)
    assert total == pytest.approx(0.775)


def test_choice_prefers_simpler_on_tie():
    assert et.choose({"norm": 0.705, "norm_lp7k": 0.70, "norm_opus": 0.62}) == "norm_opus"
    assert et.choose({"norm": 0.628, "norm_lp7k": 0.62, "norm_opus": 0.619}) == "norm"
    assert et.choose({"norm": 0.70, "norm_lp7k": 0.629, "norm_opus": 0.62}) == "norm_lp7k"


def test_cache_verification_reproduces_eligible(tmp_path):
    sel = [
        dict(id="a", parent="p", label="0"), dict(id="b", parent="p", label="1"),
        dict(id="c", parent="q", label="0"), dict(id="d", parent="q", label="1"),
    ]
    manifest = {"a": "h1", "b": "h2", "c": "h3", "d": None}
    et.verify(sel, [dict(sel[0], sha256="h1"), dict(sel[1], sha256="h2")], manifest)
    with pytest.raises(ValueError):
        et.verify(sel, [dict(sel[0], sha256="h1"), dict(sel[1], sha256="XX")], manifest)
    with pytest.raises(ValueError):
        et.verify(sel, [dict(sel[0], sha256="h1")], manifest)


def test_cached_crop_is_hash_checked(tmp_path):
    b = crop()
    (tmp_path / "x.wav").write_bytes(b)
    m = {"x": hashlib.sha256(b).hexdigest(), "y": None}
    assert et.cached(tmp_path, m, "x") == b
    assert et.cached(tmp_path, m, "y") is None
    (tmp_path / "x.wav").write_bytes(b + b"0")
    with pytest.raises(ValueError):
        et.cached(tmp_path, m, "x")


def test_frozen_record_must_match_diagnostic(tmp_path):
    from pipeline_v3.data import digest

    d = tmp_path / et.DIAG
    d.mkdir()
    (d / "protocol.json").write_text("{}")
    (d / "diagnostic.json").write_text("{}")
    rec = {"treatment": "norm", "protocol_sha256": digest(d / "protocol.json"),
           "diagnostic_sha256": digest(d / "diagnostic.json")}
    (d / "frozen.json").write_text(json.dumps(rec))
    assert et.frozen_treatment(tmp_path)["treatment"] == "norm"
    (d / "diagnostic.json").write_text('{"changed": 1}')
    with pytest.raises(ValueError):
        et.frozen_treatment(tmp_path)


def test_stage0_transformed_applies_treatment_to_cached_crops(tmp_path):
    from pipeline_v3.stage0 import transformed

    b = crop()
    (tmp_path / "x.wav").write_bytes(b)
    m = {"x": hashlib.sha256(b).hexdigest(), "y": None}
    rs = [dict(id="x", pair="phonemedf"), dict(id="y", pair="phonemedf")]
    out = dict((r["id"], v) for r, v in transformed(None, rs, "norm", (tmp_path, m)))
    assert out["x"] == et.apply(b, "norm") and out["y"] is None
    with pytest.raises(ValueError):
        list(transformed(None, [dict(id="x", pair="mlaad")], "norm", (tmp_path, m)))
