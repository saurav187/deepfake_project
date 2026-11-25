from collections import Counter

import numpy as np
import pytest

from pipeline_v3 import v2a
from pipeline_v3.audio import model_view


def test_confirmation_has_new_speakers_and_balanced_folds():
    protocol = {
        f"{s}/{g}/{i}": (str(s), g)
        for s in range(80)
        for g in ("A08", "bonafide")
        for i in range(5)
    }
    old = {str(i) for i in range(40)}
    sample = v2a.confirmation_sample(protocol, old)
    assert len(sample) == 320
    assert not old.intersection(s for s, _, _ in sample.values())
    assert set(Counter((g, k) for _, g, k in sample.values()).values()) == {32}
    assert sample == v2a.confirmation_sample(dict(reversed(protocol.items())), old)


def test_balance_and_formal_source_panels():
    rows, feats = [], {}
    for split in ("train", "select"):
        for src, systems in [
            (v2a.LSV, ("-", *v2a.FAMILIES)),
            (v2a.ASV, ("-", "A08", "A08")),
        ]:
            for i in range(10):
                sp = split + str(i)
                for j, gen in enumerate(systems):
                    rid = f"{src}-{sp}-{j}"
                    rows.append(
                        dict(
                            id=rid,
                            parent=sp if src == v2a.LSV else rid,
                            speaker=sp,
                            source=src,
                            split=split,
                            generator=gen,
                            label=str(int(gen != "-")),
                        )
                    )
                    feats[rid] = {"tilt": 1.0, "codec": "pcm"}
    rows = v2a.balance(rows)
    panels = v2a.source_panel(rows, feats)
    assert all(p["worst"] == 0.5 for p in panels.values())
    assert set(panels["pooled"]["counts"].values()) == {20}
    assert set(panels[v2a.ASV]["counts"].values()) == {10}
    assert len(v2a.balance([r for r in rows if r["generator"] != "wavegrad"])) == 40


def test_v2a_pack_binding(tmp_path):
    for name in ("lineage.csv", "confirmation.json", "index.json", "recipe.json"):
        (tmp_path / name).write_text(name)
    report = dict(
        version=v2a.VERSION,
        policy="v2a-speaker-and-parent-controls-v1",
        panels={
            k: {"worst": 0.55} for k in ("pooled", v2a.LSV, v2a.ASV, *v2a.FAMILIES)
        },
        languages_audited=["en"],
        recipe_sha256=v2a.data.digest(tmp_path / "recipe.json"),
        lineage_sha256=v2a.data.digest(tmp_path / "lineage.csv"),
        confirmation_sha256=v2a.data.digest(tmp_path / "confirmation.json"),
        **{"pass": True},
    )
    v2a.save(tmp_path / "mixture-audit.json", report)
    index = tmp_path / "index.json"
    report["index_sha256"] = v2a.data.digest(index)
    v2a.verify_published(tmp_path, report, index)
    index.write_text("changed")
    with pytest.raises(ValueError):
        v2a.verify_published(tmp_path, report, index)


def test_prepared_model_view_preserves_audited_length():
    x = np.ones(64000, dtype=np.float32)
    assert (
        len(model_view(x, {"pair": v2a.ASV, "prepared_view": "v2a-fixed-norm"}))
        == 64000
    )
