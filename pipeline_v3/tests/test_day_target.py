import numpy as np
import pytest

from pipeline_v3.day_target import FAMILIES, VIEWS, cutoff, fit_head, predict, table


def test_cutoff_handles_tied_genuine_and_strict_budget():
    scores = np.r_[np.zeros(97), 1.0, 1.0, 2.0]
    assert np.sum(scores >= cutoff(scores)) == 1
    with pytest.raises(ValueError):
        cutoff([0.9])


def test_target_requires_every_family_and_channel():
    rows = [
        dict(channel=v, generator="mlaad/" + g, label="1", speaker="a", p=0.9)
        for v in VIEWS
        for g in FAMILIES
    ]
    rows += [
        dict(channel=v, generator="genuine", label="0", speaker="b", p=0.1)
        for v in VIEWS
    ]
    assert table(rows, 0.5)["development_target_met"]
    rows[0]["p"] = 0.4
    assert not table(rows, 0.5)["development_target_met"]
    with pytest.raises(ValueError):
        table(rows[1:], 0.5)


def test_serialised_linear_head_keeps_sign():
    xs = [[1.0, 8.0], [2.0, 8.0], [7.0, 8.0], [8.0, 8.0]]
    head = fit_head(xs, [0, 0, 1, 1])
    assert predict(head, xs[0]) < 0 < predict(head, xs[-1])
