"""English-only path: one-language audit works, still enforces English gates, fails closed."""

import pytest

from pipeline_v3.audit import panels
from pipeline_v3.tests.test_audit_policy import _panel


def _english(acoustic_cue):
    rows, features = _panel(acoustic_cue)
    keep = [i for i, r in enumerate(rows) if r["language"] == "en"]
    return [rows[i] for i in keep], [features[i] for i in keep]


def test_english_only_audit_passes_without_hindi():
    rows, features = _english(acoustic_cue=False)
    with pytest.raises(ValueError):
        panels(rows, features)  # bilingual default still demands Hindi cells
    result = panels(rows, features, ["en"])
    assert result["pass"] and result["languages_audited"] == ["en"]
    assert set(result["languages"]) == {"en"}


def test_english_only_audit_still_blocks_acoustic_cue():
    rows, features = _english(acoustic_cue=True)
    result = panels(rows, features, ["en"])
    assert not result["pass"]
    assert "upper_band" in result["languages"]["en"]["failed_fields"]


def test_unconfigured_language_rows_fail_closed():
    rows, features = _panel(acoustic_cue=False)
    with pytest.raises(ValueError):
        panels(rows, features, ["en"])


def test_subset_is_deterministic_and_keeps_parents():
    from pipeline_v3.stage0 import subset

    rows = []
    for split in ("train", "select"):
        for reader in range(6):
            for k in range(10):
                parent = f"p-{split}-{reader}-{k}"
                rows.append(dict(id=parent + "-r", label="0", split=split, language="en",
                                 speaker=f"s{reader}", parent=parent, generator="-"))
                gen = "xtts_v2" if split == "train" else "chatterboxtts"
                rows.append(dict(id=parent + "-f", label="1", split=split, language="en",
                                 speaker=f"s{reader}", parent=parent, generator=gen))
    quotas = {"train": {"xtts_v2": 12}, "select": {"chatterboxtts": 6}}
    a, b = subset(rows, quotas), subset(list(reversed(rows)), quotas)
    assert {r["id"] for r in a} == {r["id"] for r in b}
    fakes = [r for r in a if r["label"] == "1"]
    assert sum(r["split"] == "train" for r in fakes) == 12
    assert len({r["speaker"] for r in fakes if r["split"] == "train"}) == 6  # round-robin
    assert {r["parent"] for r in fakes} == {r["parent"] for r in a if r["label"] == "0"}
