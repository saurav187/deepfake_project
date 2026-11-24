"""Compare trained runs on development data only (never test sets).

Usage (repo root):
  DETECTOR_AUDIT=detector/runs/audit_x1 venv/bin/python -m detector.eval.compare_dev \
      --runs detector/runs/models/final_wav2vec2-xls-r-300m_s42 detector/runs/models/x1_rate_masked_s42 \
      --out detector/runs/compare/x1_vs_v1 [--extra app_testing_dataset/*]

Scores every run on the same dev recordings: the ASVspoof 5 dev subset used in training
(400 per source/label/attack) plus ALL IndieFake dev recordings, each under three channels:
  clean    as stored
  mp3_32k  compressed upload
  rt24k    resampled 16 -> 24 -> 16 kHz only (shortcut control: a model that learned
           "stored at 24 kHz = fake" would flag genuine audio more here than on clean)
Reports, per run: EER/AUC per source and channel, and IndieFake fake detection and genuine
false alarms by stored sample rate at one common operating point (the logit at which each
run flags 5 % of the clean ASVspoof 5 + IndieFake genuine dev audio, source-balanced).
--extra files (e.g. the user's own clips) are scored and listed for information only;
they are never used to choose anything.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from detector import paths
from detector.audio import SR, decode, prepare
from detector.model.ssl import SSLDetector
from detector.train.data import dev_subset, load_manifest
from detector.train.train import dev_metrics, score_dev

CHANNELS = ("clean", "mp3_32k", "rt24k")


def operating_logit(rows, logits, fa=0.05):
    """Logit threshold at which a source-balanced fa share of clean genuine dev audio is flagged."""
    thresholds = []
    for source in ("asvspoof5", "indiefake"):
        s = np.array([x for r, x in zip(rows, logits) if r["source"] == source and r["label"] == 0 and np.isfinite(x)])
        thresholds.append(np.quantile(s, 1 - fa))
    return float(np.mean(thresholds))


def rates(rows, logits, t):
    out = {}
    for label, name in ((0, "genuine flagged"), (1, "fake detected")):
        for rate in sorted({r["rate"] for r in rows if r["source"] == "indiefake" and r["label"] == label}):
            s = np.array([x for r, x in zip(rows, logits)
                          if r["source"] == "indiefake" and r["label"] == label and r["rate"] == rate and np.isfinite(x)])
            if len(s):
                out[f"indiefake {name} @{int(rate) // 1000 if rate else '?'}k (n={len(s)})"] = float(np.mean(s >= t))
    s = np.array([x for r, x in zip(rows, logits) if r["source"] == "asvspoof5" and r["label"] == 1 and np.isfinite(x)])
    out[f"asvspoof5 fake detected (n={len(s)})"] = float(np.mean(s >= t))
    return out


@torch.inference_mode()
def score_files(model, files, device):
    out = {}
    for f in files:
        wave = prepare(decode(Path(f).read_bytes())[0])
        size = 4 * SR
        if len(wave) < size:
            wave = np.tile(wave, int(np.ceil(size / max(len(wave), 1))))[:size]
        n = int(np.ceil(len(wave) / size))
        starts = np.linspace(0, len(wave) - size, n).astype(int) if n > 1 else [0]
        x = torch.from_numpy(np.stack([wave[s:s + size] for s in starts])).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out[Path(f).name] = float(model(x).float().mean())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dev = load_manifest({"dev"})
    rows = [r for r in dev_subset(dev, 400) if r["source"] == "asvspoof5"] + [r for r in dev if r["source"] == "indiefake"]
    print(f"dev rows: {len(rows)} (audit {paths.AUDIT})")
    encoder = json.loads((args.runs[0] / "config.json").read_text())["encoder"]
    model = SSLDetector(encoder).to(device)
    summary = dict(audit=str(paths.AUDIT), rows=len(rows), runs={})
    scores = {}
    for run in args.runs:
        saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        assert saved["encoder"] == encoder, "all runs must share the encoder"
        model.load_trainable_state(saved["state"])
        model.to(device)
        res = dict(epoch=saved["epoch"], metrics=[], operating_point={})
        for channel in CHANNELS:
            logits = score_dev(model, rows, channel, device, args.workers)
            scores[(run.name, channel)] = logits
            res["metrics"] += dev_metrics(rows, logits, channel)
        t = operating_logit(rows, scores[(run.name, "clean")])
        res["operating_logit"] = t
        for channel in CHANNELS:
            res["operating_point"][channel] = rates(rows, scores[(run.name, channel)], t)
        if args.extra:
            res["extra_logits"] = score_files(model, args.extra, device)
            res["extra_flagged"] = {k: v >= t for k, v in res["extra_logits"].items()}
        summary["runs"][run.name] = res

    with (args.out / "dev_scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        keys = list(scores)
        w.writerow(["uid", "source", "label", "rate", *[f"{r}|{c}" for r, c in keys]])
        for i, r in enumerate(rows):
            w.writerow([r["uid"], r["source"], r["label"], r["rate"], *[scores[k][i] for k in keys]])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for name, res in summary["runs"].items():
        print(f"\n=== {name} (best epoch {res['epoch']}; operating logit {res['operating_logit']:.2f})")
        for m in res["metrics"]:
            print(f"  {m['channel']:8} {m['source']:10} n={m['n']:>5}  AUC {m['auc']:.4f}  EER {m['eer']:.4f}")
        for channel, d in res["operating_point"].items():
            print(f"  [{channel}] " + "; ".join(f"{k}: {v:.1%}" for k, v in d.items()))
        for k, v in res.get("extra_logits", {}).items():
            print(f"  extra {k}: logit {v:.2f} -> {'FLAGGED fake' if v >= res['operating_logit'] else 'passes as real'}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
