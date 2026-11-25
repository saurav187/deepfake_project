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
