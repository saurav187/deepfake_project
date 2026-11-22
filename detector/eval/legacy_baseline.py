"""Score the legacy CNN-LSTM models on a held-out corpus: the baseline to beat.

Usage (repo root):
  venv/bin/python -m detector.eval.legacy_baseline --source in_the_wild --out detector/runs/baselines/legacy_itw

Uses the legacy feature path unchanged (16 kHz mono, one-second log-Mel windows,
mean of non-constant window scores). Each model keeps the operating point it was
given on its own development data; nothing is tuned on the test corpus.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from detector import paths
from detector.audio import _ffmpeg_decode
from detector.data.sources import ADAPTERS, read_bytes

sys.path.insert(0, str(paths.LEGACY / "pipeline"))
import librosa  # noqa: E402
from audio_features import SR, feature, windows  # noqa: E402  (legacy feature code)
from model import CNNLSTMDetector  # noqa: E402

FROZEN = paths.LEGACY / "pipeline/outputs/audited_v1/full"
CHANNEL_RUN = paths.LEGACY / "research/channel_augmentation_v1"


def models():
    """(name, checkpoint, cutoff, cutoff description)."""
    rule = json.loads((FROZEN / "rule.json").read_text())
    out = [("audited_baseline", FROZEN / "checkpoint.pth", rule["threshold"], "frozen rule")]
    config = json.loads((CHANNEL_RUN / "config.json").read_text())
    for m in config["models"]:
        if m["arm"] != "audited_baseline":
            out.append((f"{m['arm']}_{m['seed']}", paths.LEGACY / m["checkpoint"], m["dev_cutoff"],
                        "IndieFake dev 5% FPR cutoff"))
    return out


def legacy_decode(data):
    """Same steps as the legacy read_audio, applied to bytes (including its ffmpeg fallback)."""
    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        with tempfile.NamedTemporaryFile(suffix=".audio") as tmp:
            tmp.write(data)
            tmp.flush()
            return _ffmpeg_decode(tmp.name)
    mono = audio.mean(axis=1)
    return librosa.resample(mono, orig_sr=rate, target_sr=SR) if rate != SR else mono


def features(record):
    try:
        wave = legacy_decode(read_bytes(record.locator))
    except Exception as exc:  # undecodable test files are reported, never dropped silently
        return record, None, None, f"{type(exc).__name__}: {exc}"[:200]
    feats = [x for x in (feature(w) for w in windows(wave)) if np.ptp(x) > 0]
    return record, len(wave) / SR, (np.stack(feats) if feats else np.empty((0, 128, 94), np.float32)), ""


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="in_the_wild", choices=list(ADAPTERS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="score only the first N records (smoke test)")
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = []
    for name, ckpt, cutoff, why in models():
        net = CNNLSTMDetector().to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        nets.append((name, net.eval(), cutoff, why))
    records = list(ADAPTERS[args.source]())
    if args.limit:
        records = records[: args.limit]
    start = perf_counter()
    results, pending = [], []

    @torch.inference_mode()
    def flush():
        X = torch.from_numpy(np.concatenate([f for _, _, f in pending])).unsqueeze(1)
        scores = {}
        for name, net, _, _ in nets:
            scores[name] = torch.cat([net(X[i:i + 1024].to(device)).flatten().sigmoid().float().cpu()
                                      for i in range(0, len(X), 1024)]).numpy()
        offset = 0
        for record, duration, f in pending:
            n = len(f)
            results.append(dict(uid=record.uid, label=record.label, group=record.group, duration_s=duration,
                                windows=n, error="" if n else "no non-constant windows",
                                **{name: float(s[offset:offset + n].mean()) if n else None for name, s in scores.items()}))
            offset += n
        pending.clear()

    with ThreadPoolExecutor(8) as pool:
        for i, (record, duration, f, error) in enumerate(pool.map(features, records, buffersize=64), 1):
            if error:
                results.append(dict(uid=record.uid, label=record.label, group=record.group, duration_s=None,
                                    windows=None, error=error, **{n: None for n, *_ in nets}))
            else:
                pending.append((record, duration, f))
                if sum(len(x[2]) for x in pending) >= 4096:
                    flush()
            if i % 2000 == 0:
                print(f"{i}/{len(records)} files, {perf_counter() - start:.0f}s", flush=True)
    if pending:
        flush()

    summary = []
    for name, _, cutoff, why in nets:
        scored = [r for r in results if r[name] is not None]
        y = np.array([r["label"] for r in scored])
        s = np.array([r[name] for r in scored])
        pred = s >= cutoff
        real, fake = int((y == 0).sum()), int((y == 1).sum())
        summary.append(dict(model=name, cutoff=cutoff, cutoff_from=why, scored=len(scored),
                            unscored=len(results) - len(scored), genuine=real, fake=fake,
                            auc=roc_auc_score(y, s), eer=eer(y, s),
                            false_alarms=int((pred & (y == 0)).sum()), false_alarm_rate=float((pred & (y == 0)).sum() / real),
                            detected=int((pred & (y == 1)).sum()), recall=float((pred & (y == 1)).sum() / fake)))
    for name, rows in (("predictions.csv", results), ("summary.csv", summary)):
        with (args.out / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.out / "run.json").write_text(json.dumps(dict(source=args.source, records=len(records), limit=args.limit,
        elapsed_s=perf_counter() - start, device=str(device), models=[(n, c, w) for n, _, c, w in nets]), indent=2) + "\n")
    print(f"\n{'model':22}{'AUC':>7}{'EER':>7}  false alarms        detected")
    for r in summary:
        print(f"{r['model']:22}{r['auc']:7.3f}{r['eer']:7.3f}  {r['false_alarms']:>6}/{r['genuine']:<6} ({r['false_alarm_rate']:5.1%})"
              f"  {r['detected']:>6}/{r['fake']:<6} ({r['recall']:5.1%})")


if __name__ == "__main__":
    main()
