"""Score release detector v1 on the libri-dev genuine crops (off-panel calibration genuine).

Input: a directory with manifest.json and crop/<key>.bin (the fixed 4 s crop + norm bytes
produced by pipeline_v3.clip_cache.treat on the 841 libri-dev genuine rows scored by
Teacher V1). Output: JSON {id: p_release} plus the release hash. Audio is not kept.
Run from the repository root that holds detector/ (frozen; imported only).
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crops", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from detector.inference import Detector

    from .audio import decode

    det = Detector()
    man = json.loads((a.crops / "manifest.json").read_text())
    out = {}
    for i, m in enumerate(man, 1):
        b = (a.crops / "crop" / f"{m['key']}.bin").read_bytes()
        if hashlib.sha256(b).hexdigest() != m["crop_sha256"]:
            raise ValueError(f"crop hash mismatch {m['id']}")
        r = det.analyze_wave(decode(b).astype(np.float32))
        out[m["id"]] = {"p": r["fake_probability"], "speaker": m["speaker"], "panel": m["panel"]}
        if i % 100 == 0:
            print(i, flush=True)
    res = {"release_sha256": det.release_sha256, "t_fake": det.t_fake, "scores": out}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", a.out, len(out))


if __name__ == "__main__":
    main()
