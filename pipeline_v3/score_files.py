"""Score arbitrary audio files with a frozen teacher (no fitting, no thresholds chosen here).

Uses the training preprocessing (audio.fixed_crop 4 s centre crop of the trimmed span, then
the frozen RMS norm) and the frozen Platt calibration and thresholds from teacher-frozen.json.
Also reports a full-length variant that averages the teacher over every 4 s window of the
trimmed, normalised recording, for context. Never writes into the input directory.

  python -m pipeline_v3.score_files --teacher runs/teacher.best.pt --frozen runs/teacher-frozen.json \
      --encoder models/xlsr --root app_testing_dataset --out scores.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    import torch
    from scipy.special import expit

    from .audio import decode, fixed_crop, model_view, prepare
    from .data import digest
    from .en_treatment import apply
    from .models import Teacher
    from .teacher_eval import mean_logit, verdict

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    frozen = json.loads(a.frozen.read_text())
    if digest(a.teacher) != frozen["teacher_sha256"]:
        raise ValueError("Teacher file differs from the frozen record")
    (pa, pb), (lo, hi) = frozen["platt"], frozen["thresholds"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(a.teacher, map_location="cpu", weights_only=True)
    model = Teacher(a.encoder).to(device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    rows = []
    with torch.inference_mode():
        for f in sorted(x for x in a.root.rglob("*") if x.is_file()):
            rel = str(f.relative_to(a.root))
            r = {"file": rel, "folder": str(Path(rel).parent), "seconds": None, "p_crop": None,
                 "verdict_crop": "insufficient_audio", "p_full": None, "verdict_full": "insufficient_audio",
                 "windows_full": 0, "error": ""}
            try:
                raw = f.read_bytes()
                x = decode(raw)
                r["seconds"] = round(len(x) / 16000, 2)
                crop = fixed_crop(raw)
                if crop is not None:
                    v = model_view(decode(apply(crop, "norm")), {"pair": "phonemedf"})
                    s = mean_logit(model, v, device)
                    if s is not None:
                        r["p_crop"] = float(expit(pa * s + pb))
                        r["verdict_crop"] = verdict(r["p_crop"], lo, hi)
                t, _ = prepare(x)
                if len(t) >= 16000:
                    t = 0.05 * t / max(float(np.sqrt(np.mean(t**2))), 1e-6)
                    from .audio import windows

                    ws = [w for _, w in windows(t)]
                    logits = [mean_logit(model, model_view(w, {"pair": "phonemedf"}), device) for w in ws if len(w) >= 64000]
                    logits = [z for z in logits if z is not None]
                    if logits:
                        r["windows_full"] = len(logits)
                        r["p_full"] = float(expit(pa * float(np.mean(logits)) + pb))
                        r["verdict_full"] = verdict(r["p_full"], lo, hi)
            except Exception as e:
                r["error"] = f"{type(e).__name__}: {e}"[:200]
            rows.append(r)
            print(json.dumps(r), flush=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
