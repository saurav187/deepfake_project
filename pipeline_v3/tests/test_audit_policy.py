"""Option A audit policy: provenance warns, acoustic cues block, old reports are rejected."""

import pytest

from pipeline_v3.audit import POLICY, panels, require_passing


def _panel(acoustic_cue):
    """Two languages x two source pairs x train/select x real/fake, matched parents."""
    rows, features = [], []
    for lang in ("en", "hi"):
        for pair in ("a", "b"):
            for split in ("train", "select"):
                for k in range(6):
                    for label in ("0", "1"):
                        rows.append(
                            {
                                "id": f"{lang}{pair}{split}{k}{label}",
                                "label": label,
                                "parent": f"{lang}{pair}{split}{k}",
                                "split": split,
                                "language": lang,
                                "channel": "clean",
                                "pair": pair,
                            }
                        )
                        # Provenance is perfectly predictive (different corpora per label).
                        features.append(
                            {
                                "source": f"real-{pair}" if label == "0" else f"fake-{pair}",
                                "pair": pair,
                                # Uninformative acoustic value, identical across labels...
                                "duration": 4.0 + k,
                                # ...unless the test injects a label-predictive cue.
                                "upper_band": float(label) if acoustic_cue else 0.1 * k,
                            }
                        )
    return rows, features


def test_provenance_alone_warns_but_passes():
    rows, features = _panel(acoustic_cue=False)
    result = panels(rows, features)
    assert result["policy"] == POLICY
    assert result["pass"]
    assert result["provenance_warning"]
    assert result["provenance_auc"]["source"] == 1.0
    assert "source" not in result["blocking_fields"]
    assert "pair" not in result["blocking_fields"]


def test_acoustic_cue_blocks():
    rows, features = _panel(acoustic_cue=True)
    result = panels(rows, features)
    assert not result["pass"]
    assert "upper_band" in result["failed_fields"]
    assert result["languages"]["hi"]["failed_fields"]  # reported per panel too


def test_joint_excludes_provenance():
    rows, features = _panel(acoustic_cue=False)
    joint = panels(rows, features)["controls"]["joint"]["auc"]
    assert max(joint, 1 - joint) <= 0.60


def test_old_policy_cannot_authorize():
    good = {"policy": POLICY, "pass": True, "index_sha256": "x"}
    require_passing(good, "x")
    for bad in (
        dict(good, policy="source-strict-v1"),
        dict(good, policy=None),
        dict(good, **{"pass": False}),
        dict(good, index_sha256="other"),
    ):
        with pytest.raises(ValueError):
            require_passing(bad, "x")
