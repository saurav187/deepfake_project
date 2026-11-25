import numpy as np

from pipeline_v3.phonemedf import REFS
from pipeline_v3.voice_ecapa import classify_rule


def test_rule_requires_agreement_floor_and_margin():
    A = np.eye(10, 16)
    good = np.array([A[3] * 0.8 + A[5] * 0.1, A[3] * 0.7])
    assert classify_rule(good, A, 0.64, 0.08)[0] == REFS[3]
    assert classify_rule(good, A, 0.75, 0.08)[0] == "unknown"  # below floor on one view
    split = np.array([A[3] * 0.8, A[4] * 0.8])
    assert classify_rule(split, A, 0.64, 0.08)[0] == "unknown"  # views disagree
    close = np.array([A[3] * 0.8 + A[4] * 0.75, A[3] * 0.8])
    assert classify_rule(close, A, 0.64, 0.08)[0] == "unknown"  # margin too small
