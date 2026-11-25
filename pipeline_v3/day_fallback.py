"""Day-target fallback: the released three-head detector v1, frozen, on the same MLAAD trials.

Scores exactly the (id, channel) trials of the V1 teacher reference that day_target.evaluate
scores. Weights, calibration and the likely_fake threshold come unchanged from
detector/runs/release/v1 (built at a 5% dev false-alarm budget, not 2%). No fitting.

Two views per trial:
  paired : the same fixed 4 s crop + norm and simulated channel as the day-target candidates
  full   : the whole recording through the same simulated channel (the detector's own use)
The paired view gives the headline table; full is descriptive.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .day_target import VIEWS, reference, table
from .v2a import V1, save

OUT = "fallback-detector-v1"
# Local source copy detector/pretrained/wav2vec2-xls-r-300m, the encoder the heads were trained with.
ENCODER_SHA256 = {
    "config.json": "0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac",
    "preprocessor_config.json": "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335",
    "pytorch_model.bin": "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0",
}


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--provenance", type=Path, required=True,
                   help="JSON with the release audit manifest's train/dev PCM hashes")
    a = p.parse_args()

    from detector.audio import decode as det_decode
    from detector.audio import pcm_hash
    from detector.inference import Detector
    from detector.model.ssl import ENCODERS

    from .adapters import selected_members
    from .audio import channel, decode
    from .clip_cache import treat
    from .stage0 import groups
    from .teacher_eval import panel_rows

    root = a.data
    work = root / "v2/day-target-1" / OUT
    work.mkdir(parents=True, exist_ok=True)
    enc = Path(ENCODERS["wav2vec2-xls-r-300m"])
    for name, want in ENCODER_SHA256.items():
        if file_sha(enc / name) != want:
            raise ValueError(f"Encoder file differs from the training copy: {name}")
    det = Detector()
    prov = json.loads(a.provenance.read_text())
    if prov["provenance"]["audit_manifest_sha256"] != det.release["audit_manifest_sha256"]:
        raise ValueError("Provenance is not from the release's audit manifest")
    trained = set(prov["hashes"])
    original = reference(root / V1)
    ref = {(r["id"], r["channel"]): r for r in original}
    needed = {uid for uid, _ in ref}
    rows = [r for r in panel_rows(root, ["mlaad"]) if r["id"] in needed]
    if {r["id"] for r in rows} != needed or len(rows) != len(needed):
        raise ValueError("Reference panel IDs no longer resolve")
    binding = {
        "release_sha256": det.release_sha256,
        "members": [m["checkpoint_sha256"] for m in det.release["members"]],
        "threshold_likely_fake": det.t_fake,
        "encoder_sha256": ENCODER_SHA256,
        "teacher_scores": file_sha(root / V1 / "runs/teacher-eval-clips.csv"),
        "provenance": prov["provenance"],
        "provenance_file_sha256": file_sha(a.provenance),
    }
    bpath = work / "inputs.json"
    if bpath.exists() and json.loads(bpath.read_text()) != binding:
        raise ValueError("Fallback inputs changed")
    save(bpath, binding)
    scores = work / "scores"
    scores.mkdir(exist_ok=True)

    def output(uid):
        return scores / (hashlib.sha256(uid.encode()).hexdigest() + ".json")

    def one(x):
        r = det.analyze_wave(x.astype(np.float32))
        return {"p": r["fake_probability"], "verdict": r["verdict"], "speech_s": r["speech_s"]}

    pending = [r for r in rows if not output(r["id"]).exists()]
    for i, (r, raw) in enumerate(selected_members(groups(root / "development", pending)), 1):
        b = treat(raw)
        if b is None:
            raise ValueError("Previously eligible comparison clip failed")
        crop, full = decode(b), decode(raw)
        rec = {"_overlap": pcm_hash(det_decode(raw)[0]) in trained}
        for view in VIEWS:
            if (r["id"], view) in ref:
                rec[view] = {"paired": one(channel(crop, view)), "full": one(channel(full, view))}
        save(output(r["id"]), rec)
        if i % 100 == 0:
            print(f"Fallback panel {i}/{len(pending)}", flush=True)

    report, excluded, overlap = {}, defaultdict(int), []
    for row in rows:
        if json.loads(output(row["id"]).read_text())["_overlap"]:
            overlap.append(row["id"])
    for kind in ("paired", "full"):
        scored = []
        for row in rows:
            rec = json.loads(output(row["id"]).read_text())
            if rec.pop("_overlap"):
                continue  # exact training/dev content: excluded and logged
            for view, v in rec.items():
                s = v[kind]
                if s["p"] is None:  # insufficient audio: counted as not flagged, logged
                    excluded[f"{kind}/{view}/{ref[row['id'], view]['label']}"] += 1
                scored.append(dict(ref[row["id"], view], p=-1.0 if s["p"] is None else s["p"]))
        if len(scored) + sum(
            (o, v) in ref for o in overlap for v in VIEWS
        ) != len(ref):
            raise ValueError("Incomplete fallback scores")
        report[kind] = table(scored, det.t_fake)
    report.update(
        inputs=binding,
        exact_training_overlap_excluded=overlap,
        insufficient_audio_counted_not_flagged=dict(excluded),
        operating_point=f"release v1 likely_fake p >= {det.t_fake:.4f} (5% FA / 5% miss dev budget, frozen)",
        scope="Previously inspected English MLAAD development panel, simulated channels. Real calls unmeasured.",
    )
    save(work / "result.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
