"""Codec probe: matched-pair selection keeps folds/pairs; treatments apply to both labels."""

import io
import wave

import numpy as np

from pipeline_v3.codec_probe import choose, probe, treat


def test_choose_matched_pairs_per_fold():
    rows = []
    for split in ("train", "select", "validate"):
        for k in range(4):
            parent = f"p-{split}-{k}"
            rows.append({"id": f"r{split}{k}", "label": "0", "parent": parent, "split": split, "language": "hi"})
            for g in ("a", "b"):
                rows.append({"id": f"f{split}{k}{g}", "label": "1", "parent": parent, "split": split, "language": "hi"})
    picked = choose(rows, per_split=3)
    fakes = [r for r in picked if r["label"] == "1"]
    assert {r["split"] for r in picked} == {"train", "select"}  # no evaluation folds
    assert len(fakes) == 6 and len({r["parent"] for r in fakes}) == 6  # one fake per parent
    reals = {r["parent"] for r in picked if r["label"] == "0"}
    assert reals == {r["parent"] for r in fakes}  # every fake has its matched real


def test_treatments_reencode():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes((np.sin(np.arange(22050 * 6) / 7) * 9000).astype("<i2").tobytes())
    b = buf.getvalue()
    assert probe(b)["codec"] == "pcm_s16le"
    assert probe(treat(b, "aac"))["codec"] == "aac"
    assert probe(treat(b, "opus"))["codec"] == "opus"
    for name in ("crop", "crop_norm", "crop_norm_lp7k"):
        assert probe(treat(b, name))["codec"] == "pcm_s16le"
    assert probe(treat(b, "crop_norm_opus"))["codec"] == "opus"
