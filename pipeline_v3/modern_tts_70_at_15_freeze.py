"""Freeze the combined candidate for independent calibration/evaluation.

Rule (prespecified in modern_tts_70_at_15.py, weight chosen as 0.5 in all six
reader-disjoint rotations on the development panel):

    z_m   = (logit(clip(p_m, 1e-6, 1-1e-6)) - mu_m) / sd_m      m in {release, teacher}
    score = 0.5 * z_release + 0.5 * z_teacher                   flag if score >= cutoff

p_release : release detector v1 (detector/runs/release/v1, sha bdb7748c...) fake_probability
            on the "paired" view: the fixed 4 s crop + frozen "norm" treatment
            (pipeline_v3.clip_cache.treat), clean channel.
p_teacher : exploratory Teacher V1 (teacher sha 8b4625fb...) Platt probability on the same crop.

mu/sd are fitted here on the 697 genuine clips of the 3-reader MLAAD development panel
(fit data only). The cutoff is NOT set here: it must come from a separate calibration
manifest's genuine clips (at most 15% flagged) and be frozen before evaluation.
"""

import argparse
import json
import statistics
from pathlib import Path

from .modern_tts_70_at_15 import load, logit, sha, TEACHER_CSV, FALLBACK

WEIGHT_RELEASE = 0.5


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = load(a.data)
    gen = [r for r in rows if r["label"] == 0]
    stats = {}
    for m in ("release", "teacher"):
        v = [logit(r[m]) for r in gen]
        stats[m] = {"mu": statistics.fmean(v), "sd": statistics.pstdev(v)}
    frozen = {
        "candidate": "release-v1 x teacher-v1 z-mean (w_release=0.5)",
        "weight_release": WEIGHT_RELEASE,
        "logit_clip_eps": 1e-6,
        "normaliser": stats,
        "normaliser_fit": f"{len(gen)} genuine clips, 3-reader MLAAD development panel, clean, paired view",
        "inputs": {
            "release": "detector release v1 fake_probability, paired view (fixed 4 s crop + norm), clean",
            "teacher": "Teacher V1 Platt p on the same crop, clean",
        },
        "release_sha256": json.loads((a.data / FALLBACK / "inputs.json").read_text())["release_sha256"],
        "teacher_sha256": json.loads(
            (a.data / "stage0-en-ecapa-xtts-norm-exploratory/runs/teacher-frozen.json").read_text()
        )["teacher_sha256"],
        "teacher_csv_sha256": sha(a.data / TEACHER_CSV),
        "cutoff": "unset: choose on separate calibration genuine (<=15% flagged), freeze before evaluation",
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "frozen-candidate.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
