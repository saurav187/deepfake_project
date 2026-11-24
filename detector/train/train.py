"""Train Model A (frozen SSL encoder + head) on the audited train split.

Usage (repo root):
  venv/bin/python -m detector.train.train --encoder wavlm-base-plus --seed 42 \
      --out detector/runs/models/wavlm_s42

Selection uses dev only: after every epoch, dev recordings are scored clean and
through MP3 32 kb/s, and the epoch with the lowest mean per-source EER is kept.
Test corpora (In-the-Wild, VA) are never loaded here.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from detector import paths
from detector.model.ssl import ENCODERS, SSLDetector
from detector.train.data import (BalancedCellSampler, DevWindows, TrainWindows, collate_dev,
                                 dev_subset, load_manifest)

DEV_CHANNELS = ("clean", "mp3_32k")
MAX_LOAD_FAILURE_FRACTION = 0.01
# Python 3.14 defaults to forkserver, where every worker re-imports torch and
# transformers (~1 GB each) and the run was OOM-killed on a 14 GB machine.
# Workers only decode/augment audio on the CPU and never touch CUDA, so fork
# is safe and shares the parent's memory copy-on-write.
WORKER_CONTEXT = "fork"


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def eer(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    return float((fpr[i] + 1 - tpr[i]) / 2)


@torch.inference_mode()
def score_dev(model, rows, channel, device, workers, batch_recordings=8):
    loader = DataLoader(DevWindows(rows, channel), batch_size=batch_recordings, num_workers=workers,
                        collate_fn=collate_dev, persistent_workers=False, multiprocessing_context=WORKER_CONTEXT)
    sums, counts = np.zeros(len(rows)), np.zeros(len(rows))
    model.eval()
    for windows, owners in loader:
        if len(windows):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(windows.to(device, non_blocking=True)).float().cpu().numpy()
            # owners are global recording indices (DevWindows returns its dataset index)
            np.add.at(sums, owners.numpy(), logits)
            np.add.at(counts, owners.numpy(), 1)
    with np.errstate(invalid="ignore"):
        return sums / counts  # mean logit per recording; NaN where decoding failed


def dev_metrics(rows, logits, channel):
    out = []
    by_source = defaultdict(list)
    for r, s in zip(rows, logits):
        if np.isfinite(s):
            by_source[r["source"]].append((r["label"], s))
    for source, items in sorted(by_source.items()):
        y, s = np.array([i[0] for i in items]), np.array([i[1] for i in items])
        if len(set(y)) == 2:
            out.append(dict(channel=channel, source=source, n=len(y), auc=float(roc_auc_score(y, s)), eer=eer(y, s)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", choices=list(ENCODERS), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--steps", type=int, default=600, help="optimizer steps per epoch")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=4, help="stop after this many epochs without dev improvement")
    p.add_argument("--dev-per-cell", type=int, default=400, help="dev recordings per (source, label, attack)")
    p.add_argument("--workers", type=int, default=min(6, max(2, (os.cpu_count() or 4) - 2)))
    p.add_argument("--rate-roundtrip-p", type=float, default=0.0,
                   help="probability of a random sample-rate round trip per training recording (both labels)")
    p.add_argument("--lowpass-p", type=float, default=0.0,
                   help="probability of a random 4-7.5 kHz low-pass per training window (both labels)")
    p.add_argument("--smoke", action="store_true", help="tiny run to test the pipeline end to end")
    args = p.parse_args()
    if args.smoke:
        args.epochs, args.steps, args.dev_per_cell, args.patience = 2, 10, 8, 2
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = load_manifest({"train"})
    dev_rows = dev_subset(load_manifest({"dev"}), args.dev_per_cell)
    cells = Counter((r["source"], r["label"]) for r in train_rows)
    print(f"train recordings by (source, label): {dict(cells)}")
    print(f"dev recordings scored: {len(dev_rows)} ({dict(Counter(r['source'] for r in dev_rows))})")

    model = SSLDetector(args.encoder).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * args.steps,
                                                    pct_start=0.1)
    loss_fn = torch.nn.BCEWithLogitsLoss()  # classes are balanced by the sampler
    sampler = BalancedCellSampler(train_rows, args.steps, args.batch_size, args.seed)
    loader = DataLoader(TrainWindows(train_rows, rate_roundtrip_p=args.rate_roundtrip_p, lowpass_p=args.lowpass_p), batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4,
                        multiprocessing_context=WORKER_CONTEXT)

    config = dict(vars(args), out=str(args.out), device=str(device), python=platform.python_version(),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                  trainable_params=sum(p.numel() for p in trainable),
                  train_cells={f"{s}/{l}": n for (s, l), n in sorted(cells.items())},
                  dev_recordings=len(dev_rows), dev_channels=DEV_CHANNELS,
                  audit_report=str(paths.AUDIT / "report.json"),
                  # Freeze the exact inventory: later audit reruns must not change what this run claims.
                  audit_report_sha256=file_sha256(paths.AUDIT / "report.json"),
                  audit_manifest_sha256=file_sha256(paths.AUDIT / "manifest.csv"),
                  train_uids_sha256=hashlib.sha256("\n".join(sorted(r["uid"] for r in train_rows)).encode()).hexdigest(),
                  dev_uids_sha256=hashlib.sha256("\n".join(r["uid"] for r in dev_rows).encode()).hexdigest())
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    history, best, bad = [], np.inf, 0
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        t0, losses, failures, seen = perf_counter(), [], 0, Counter()
        for wave, y, index in loader:
            ok = index >= 0  # a failed load returns index -1; it must not enter the loss
            failures += int((~ok).sum())
            if not ok.any():
                continue
            wave, y = wave[ok], y[ok]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(wave.to(device, non_blocking=True))
            loss = loss_fn(logits.float(), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            seen.update(y.long().tolist())
        train_s = perf_counter() - t0
        if failures > MAX_LOAD_FAILURE_FRACTION * args.steps * args.batch_size:
            raise SystemExit(f"{failures} training windows failed to load in epoch {epoch}; check the data.")
        t1 = perf_counter()
        metrics, predictions = [], {}
        for channel in DEV_CHANNELS:
            logits = score_dev(model, dev_rows, channel, device, args.workers)
            predictions[channel] = logits
            metrics.extend(dev_metrics(dev_rows, logits, channel))
        selection = float(np.mean([m["eer"] for m in metrics]))
        layer_weights = torch.softmax(model.layer_logits.detach().float(), 0).cpu().numpy()
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), train_s=round(train_s, 1),
                   dev_s=round(perf_counter() - t1, 1), load_failures=failures, selection_eer=selection,
                   lr=scheduler.get_last_lr()[0], top_layer=int(layer_weights.argmax()),
                   **{f"{m['channel']}/{m['source']}/eer": m["eer"] for m in metrics},
                   **{f"{m['channel']}/{m['source']}/auc": m["auc"] for m in metrics})
        history.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if selection < best:
            best, bad = selection, 0
            torch.save(dict(encoder=args.encoder, epoch=epoch, state=model.trainable_state(),
                            layer_weights=layer_weights.tolist()), args.out / "best.pt")
            with (args.out / "dev_predictions.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["uid", "source", "label", "attack", *[f"logit_{c}" for c in DEV_CHANNELS]])
                for i, r in enumerate(dev_rows):
                    w.writerow([r["uid"], r["source"], r["label"], r["attack"],
                                *[predictions[c][i] for c in DEV_CHANNELS]])
        else:
            bad += 1
        with (args.out / "history.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[-1]))
            w.writeheader()
            w.writerows(history)
        if bad >= args.patience:
            print(f"early stop: no dev improvement for {args.patience} epochs")
            break
    summary = dict(best_selection_eer=best, best_epoch=min(history, key=lambda h: h["selection_eer"])["epoch"],
                   epochs_run=len(history))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
