"""Train native-format CNN-LSTMs with and without simulated call-channel augmentation.

Both arms use the identical native 16 kHz mono train cohort, model, optimizer,
shuffle order and step count. The augmented arm replaces each training window,
independently per epoch, with one of three versions of its recording: original,
8 kHz linear PCM or 8 kHz G.711 mu-law (each returned to 16 kHz). Augmentation
is applied to genuine and fake recordings alike, so the channel cannot predict
the label. Training uses a soxr/numpy channel; evaluation uses ffmpeg, so the
dev comparison is not scored on the exact transformation seen in training.

Stage 1 scores the 734 native-format dev recordings through all three channels.
Stage 2 (--asvspoof) scores ASVspoof 2021 LA eval with every model in one pass.
No ASVspoof label is used for training, threshold choice or model selection.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import tarfile
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import soxr
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

from audio_features import BASE, ROOT, RUN, SR, feature, read_audio, windows
from model import CNNLSTMDetector
from prepare_manifest import digest, write_csv

baseline = importlib.import_module("02_train_cnn_lstm")
formats = importlib.import_module("compare_training_formats")
asv = importlib.import_module("evaluate_asvspoof_la")

SEEDS = (42, 43, 44)
EPOCHS = 5
BUDGET = 0.05
CHANNELS = ("original", "float_pcm_8khz", "mulaw_8khz")
PREVIOUS = BASE / "research/training_formats_v1"
FROZEN = RUN / "full"


# ---------------------------------------------------------------- channels
def mulaw_codec(x):
    """8-bit mu-law (mu=255) quantization of samples in [-1, 1]."""
    x = np.clip(x, -1, 1)
    y = np.sign(x) * np.log1p(255 * np.abs(x)) / np.log1p(255)
    q = np.round((y + 1) * 127.5) / 127.5 - 1
    return (np.sign(q) * np.expm1(np.abs(q) * np.log1p(255)) / 255).astype(np.float32)


def train_channel(audio, channel):
    """Training-time channel simulation (soxr resampling, numpy companding)."""
    if channel == "original":
        return audio
    narrow = soxr.resample(audio, SR, 8000, quality="HQ")
    if channel == "mulaw_8khz":
        narrow = mulaw_codec(narrow)
    wide = soxr.resample(narrow, 8000, SR, quality="HQ").astype(np.float32)
    return fit_length(wide, len(audio))


def fit_length(signal, n):
    if abs(len(signal) - n) > 2:
        raise ValueError(f"Channel changed length {n} -> {len(signal)}")
    return np.pad(signal, (0, max(0, n - len(signal))))[:n]


def eval_channel(audio, channel):
    """Evaluation-time channel through ffmpeg, as in the call-channel probe."""
    if channel == "original":
        return audio
    def run(data, fmt_in, rate_in, fmt_out, rate_out):
        return subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", fmt_in, "-ar", str(rate_in),
                               "-ac", "1", "-i", "pipe:0", "-f", fmt_out, "-ar", str(rate_out),
                               "-ac", "1", "pipe:1"], input=data, capture_output=True, check=True).stdout
    fmt = {"float_pcm_8khz": "f32le", "mulaw_8khz": "mulaw"}[channel]
    encoded = run(audio.astype("<f4", copy=False).tobytes(), "f32le", SR, fmt, 8000)
    decoded = np.frombuffer(run(encoded, fmt, 8000, "f32le", SR), dtype="<f4")
    if not np.isfinite(decoded).all():
        raise ValueError("Non-finite channel output")
    return fit_length(decoded, len(audio))


# ---------------------------------------------------------------- features
def chunk_features(rows, convert, channels, verify):
    """Return {channel: (features[n,128,94], kept_mask[n])} in row order."""
    files = baseline.group_files(rows)
    position = {int(r["row_id"]): i for i, r in enumerate(rows)}
    out = {c: (np.empty((len(rows), 128, 94), np.float32), np.ones(len(rows), bool)) for c in channels}

    def load(items):
        audio = read_audio(ROOT / items[0]["source_file"])
        result = []
        for channel in channels:
            signal = convert(audio, channel)
            for row in items:
                start = int(row["chunk"]) * SR
                part = signal[start:start + SR]
                assert len(part) == int(row["samples"]), row
                x = feature(part)
                if channel == "original" and verify:
                    assert digest(x) == row["feature_hash"], row
                result.append((channel, position[int(row["row_id"])], x))
        return result

    with ThreadPoolExecutor(6) as pool:
        for items in pool.map(load, files.values(), buffersize=24):
            for channel, i, x in items:
                out[channel][0][i] = x
                out[channel][1][i] = np.ptp(x) > 0
    return out


# ---------------------------------------------------------------- training
def fit(X, rows, arm, seed, device, out):
    """X has shape (variants, n, 128, 94). The clean arm only reads variant 0."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n = len(rows)
    y = torch.tensor([float(r["label"]) for r in rows])
    ids = torch.tensor([int(r["row_id"]) for r in rows])
    # Same sampler, dataset length and generator as compare_training_formats, so
    # the clean arm reproduces its native runs and both arms share batch order.
    loader = DataLoader(TensorDataset(torch.arange(n), y, ids), batch_size=96, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    variant_rng = np.random.default_rng(10_000 + seed)
    model = CNNLSTMDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.0001)
    pos = sum(r["label"] == "0" for r in rows) / sum(r["label"] == "1" for r in rows)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos], device=device))
    usage = np.zeros(len(CHANNELS), int)
    epochs = []
    for epoch in range(1, EPOCHS + 1):
        start = perf_counter()
        variant = (variant_rng.integers(0, len(CHANNELS), n) if arm == "channel_aug"
                   else np.zeros(n, int))
        usage += np.bincount(variant, minlength=len(CHANNELS))
        variant = torch.from_numpy(variant)
        model.train()
        seen = []
        for idx, target, batch_ids in loader:
            x = X[variant[idx], idx].unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = loss_fn(model(x.to(device)).flatten(), target.to(device))
            loss.backward()
            optimizer.step()
            seen.extend(batch_ids.tolist())
        baseline.verify_ids(seen, rows)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epochs.append(perf_counter() - start)
        print(f"{arm}_{seed} epoch {epoch}/{EPOCHS}: {epochs[-1]:.2f}s", flush=True)
    torch.save(model.state_dict(), out / "checkpoint.pth")
    model.eval()
    return model, epochs, usage


@torch.inference_mode()
def chunk_scores(model, X, device, batch=512):
    values = []
    for start in range(0, len(X), batch):
        x = torch.from_numpy(X[start:start + batch]).unsqueeze(1).to(device)
        values.append(model(x).flatten().sigmoid().float().cpu().numpy())
    return np.concatenate(values) if values else np.empty(0)


def recording_scores(rows, scores, kept):
    by_file = defaultdict(list)
    for r, s, k in zip(rows, scores, kept):
        by_file[r["source_file"], r["speaker_group"], int(r["label"])].append(s if k else None)
    result = []
    for (name, speaker, label), values in by_file.items():
        values = [v for v in values if v is not None]
        result.append(dict(file=name, speaker=speaker, label=label,
                           score=float(np.mean(values)) if values else None))
    return result


def budget_cutoff(genuine_scores):
    """Smallest cutoff giving at most floor(BUDGET * n) genuine false alarms."""
    genuine = sorted(genuine_scores, reverse=True)
    return float(np.nextafter(genuine[int(BUDGET * len(genuine))], np.inf))


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def operating(y, s, cutoff):
    y, s = np.asarray(y), np.asarray(s)
    pred = s >= cutoff
    real, fake = int((y == 0).sum()), int((y == 1).sum())
    fp, tp = int((pred & (y == 0)).sum()), int((pred & (y == 1)).sum())
    return dict(real=real, fake=fake, fp=fp, tp=tp, fpr=fp / real if real else None,
                recall=tp / fake if fake else None)


def dev_metrics(items, arm, seed, clean_cutoff):
    rows = []
    for channel in CHANNELS:
        group = [r for r in items if r["channel"] == channel and r["score"] is not None]
        y = [r["label"] for r in group]
        s = [r["score"] for r in group]
        own = budget_cutoff([v for v, t in zip(s, y) if t == 0])
        at_clean = operating(y, s, clean_cutoff)
        at_own = operating(y, s, own)
        rows.append(dict(arm=arm, seed=seed, channel=channel, scored=len(group),
                         unscored=sum(r["score"] is None for r in items if r["channel"] == channel),
                         auc=roc_auc_score(y, s), eer=eer(y, s), clean_cutoff=clean_cutoff,
                         clean_cutoff_fp=at_clean["fp"], clean_cutoff_fpr=at_clean["fpr"],
                         clean_cutoff_tp=at_clean["tp"], clean_cutoff_recall=at_clean["recall"],
                         own_cutoff=own, own_cutoff_fp=at_own["fp"], own_cutoff_tp=at_own["tp"],
                         own_cutoff_recall=at_own["recall"], real=at_clean["real"], fake=at_clean["fake"]))
    return rows


def load_frozen(device):
    rule = json.loads((FROZEN / "rule.json").read_text())
    assert baseline.sha(FROZEN / "checkpoint.pth") == rule["checkpoint_sha256"]
    model = CNNLSTMDetector().to(device)
    model.load_state_dict(torch.load(FROZEN / "checkpoint.pth", map_location=device, weights_only=True))
    return model.eval(), rule


def dev_stage(args, device):
    t0 = perf_counter()
    rows = baseline.read_manifest(RUN / "manifest.csv")
    cohorts, dev, _ = formats.select_cohorts(rows)
    train = cohorts["native"]
    # Membership must equal the native arm of the earlier comparison.
    with (PREVIOUS / "membership.csv").open() as f:
        previous_ids = sorted(int(i) for r in csv.DictReader(f) if r["arm"] == "native"
                              for i in r["row_ids"].split())
    assert previous_ids == sorted(int(r["row_id"]) for r in train)
    timing = {}
    start = perf_counter()
    feats = chunk_features(train, train_channel, CHANNELS, verify=True)
    X = torch.from_numpy(np.stack([feats[c][0] for c in CHANNELS]))
    timing["train_features_s"] = perf_counter() - start
    start = perf_counter()
    dev_feats = chunk_features(dev, eval_channel, CHANNELS, verify=True)
    timing["dev_features_s"] = perf_counter() - start
    del feats

    def score_dev(model, arm, seed):
        items = []
        for channel in CHANNELS:
            Xc, kept = dev_feats[channel]
            for r in recording_scores(dev, chunk_scores(model, Xc, device), kept):
                items.append(dict(arm=arm, seed=seed, channel=channel, **r))
        clean = [r["score"] for r in items if r["channel"] == "original" and r["label"] == 0]
        return items, budget_cutoff(clean)

    metrics, predictions, timings, models = [], [], [], []
    frozen, rule = load_frozen(device)
    items, cutoff = score_dev(frozen, "audited_baseline", 42)
    metrics += dev_metrics(items, "audited_baseline", 42, cutoff)
    predictions += items
    models.append(dict(arm="audited_baseline", seed=42, checkpoint=str((FROZEN / "checkpoint.pth").relative_to(BASE)),
                       dev_cutoff=cutoff, frozen_threshold=rule["threshold"]))
    # The frozen baseline's clean dev scores must reproduce the saved within-format scores.
    with (BASE / "research/asvspoof_la_v1/matched_format_dev.csv").open() as f:
        saved = {r["file"]: float(r["score"]) for r in csv.DictReader(f)}
    replay = max(abs(saved[r["file"]] - r["score"]) for r in items if r["channel"] == "original")
    checks = dict(baseline_dev_max_abs_diff=replay)

    with (PREVIOUS / "dev_predictions.csv").open() as f:
        prior = {(int(r["seed"]), r["file"]): float(r["score"]) for r in csv.DictReader(f) if r["arm"] == "native"}
    for arm in ("native_clean", "channel_aug"):
        for seed in SEEDS:
            out = args.out / f"{arm}_{seed}"
            out.mkdir()
            model, epochs, usage = fit(X, train, arm, seed, device, out)
            timings.append(dict(arm=arm, seed=seed, training_s=sum(epochs),
                                **{f"windows_{c}": int(u) for c, u in zip(CHANNELS, usage)}))
            items, cutoff = score_dev(model, arm, seed)
            if arm == "native_clean":
                checks[f"native_clean_{seed}_vs_training_formats_v1_max_abs_diff"] = max(
                    abs(prior[seed, r["file"]] - r["score"]) for r in items if r["channel"] == "original")
            metrics += dev_metrics(items, arm, seed, cutoff)
            predictions += items
            models.append(dict(arm=arm, seed=seed, checkpoint=str((out / "checkpoint.pth").relative_to(BASE)),
                               dev_cutoff=cutoff, checkpoint_sha256=baseline.sha(out / "checkpoint.pth")))
            print(json.dumps([m for m in metrics if m["arm"] == arm and m["seed"] == seed], indent=1), flush=True)
    write_csv(args.out / "dev_metrics.csv", metrics)
    write_csv(args.out / "dev_predictions.csv", predictions)
    write_csv(args.out / "timing.csv", timings)
    config = dict(script_sha256=baseline.sha(Path(__file__)), manifest_sha256=baseline.sha(RUN / "manifest.csv"),
                  seeds=SEEDS, epochs=EPOCHS, fpr_budget=BUDGET, channels=CHANNELS,
                  train_recordings=len({r["source_file"] for r in train}), train_windows=len(train),
                  dev_recordings=len({r["source_file"] for r in dev}), dev_windows=len(dev),
                  torch=str(torch.__version__), soxr=soxr.__version__, device=str(device),
                  gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                  ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
                  checks=checks, models=models, timing=dict(timing, total_s=perf_counter() - t0))
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(checks, indent=1))


# ---------------------------------------------------------------- ASVspoof
def asv_features(raw, uid):
    audio = asv.decode_flac(raw, uid)
    feats, samples = [], 0
    for part in windows(audio):
        x = feature(part)
        if np.ptp(x) > 0:
            feats.append(x)
    return len(audio) / SR, np.stack(feats) if feats else np.empty((0, 128, 94), np.float32)


def asvspoof_stage(args, device):
    t0 = perf_counter()
    config = json.loads((args.out / "config.json").read_text())
    with asv.AUDIO.open("rb") as f:
        if hashlib.file_digest(f, "md5").hexdigest() != asv.MD5:
            raise ValueError("Archive differs from the published Zenodo MD5")
    models = []
    for m in config["models"]:
        net = CNNLSTMDetector().to(device)
        net.load_state_dict(torch.load(BASE / m["checkpoint"], map_location=device, weights_only=True))
        models.append((f"{m['arm']}_{m['seed']}", net.eval()))
    trials = asv.protocol()
    results, files = [], 0

    def work(item):
        uid, raw = item
        try:
            return uid, *asv_features(raw, uid), ""
        except (ValueError, subprocess.CalledProcessError) as exc:
            return uid, None, None, str(exc)

    def entries():
        nonlocal files
        with tarfile.open(asv.AUDIO, mode="r|gz") as archive:
            for entry in archive:
                if not entry.isfile() or not entry.name.endswith(".flac"):
                    continue
                files += 1
                uid = Path(entry.name).stem
                if uid in trials:
                    yield uid, archive.extractfile(entry).read()
                if args.limit and files >= args.limit:
                    return

    pending = []
    def flush():
        X = np.concatenate([f for _, _, f in pending])
        with torch.inference_mode():
            per_model = {}
            for name, net in models:
                per_model[name] = chunk_scores(net, X, device, batch=1024)
        offset = 0
        for uid, duration, f in pending:
            n = len(f)
            row = dict(utterance=uid, codec=trials[uid]["codec"], label=trials[uid]["label"],
                       attack=trials[uid]["attack"], duration_s=duration, windows=n, error="")
            for name in per_model:
                row[name] = float(per_model[name][offset:offset + n].mean()) if n else None
            offset += n
            results.append(row)
        pending.clear()

    with ThreadPoolExecutor(8) as pool:
        for uid, duration, f, error in pool.map(work, entries(), buffersize=64):
            if error:
                results.append(dict(utterance=uid, codec=trials[uid]["codec"], label=trials[uid]["label"],
                                    attack=trials[uid]["attack"], duration_s=None, windows=None, error=error,
                                    **{name: None for name, _ in models}))
                continue
            pending.append((uid, duration, f))
            if sum(len(p[2]) for p in pending) >= 4096:
                flush()
            if len(results) % 2000 < 1 and results:
                print(f"{len(results)} trials, {files} archive files, {perf_counter()-t0:.0f}s", flush=True)
    if pending:
        flush()
    if not args.limit and trials.keys() - {r["utterance"] for r in results}:
        raise ValueError("Archive did not cover requested trials")

    cutoffs = {f"{m['arm']}_{m['seed']}": m["dev_cutoff"] for m in config["models"]}
    frozen_threshold = config["models"][0]["frozen_threshold"]
    summary = []
    for name, _ in models:
        rules = [("dev_budget_cutoff", cutoffs[name])]
        if name == "audited_baseline_42":
            rules.append(("frozen_rule", frozen_threshold))
        for codec in ("none", "ulaw"):
            group = [r for r in results if r["codec"] == codec and r[name] is not None]
            y = [int(r["label"] == "spoof") for r in group]
            s = [r[name] for r in group]
            for rule_name, cutoff in rules:
                summary.append(dict(model=name, codec=codec, rule=rule_name, cutoff=cutoff,
                                    unscored=sum(1 for r in results if r["codec"] == codec and r[name] is None),
                                    auc=roc_auc_score(y, s), eer=eer(y, s), **operating(y, s, cutoff)))
    write_csv(args.out / "asvspoof_predictions.csv", results)
    write_csv(args.out / "asvspoof_summary.csv", summary)
    (args.out / "asvspoof_run.json").write_text(json.dumps(dict(
        archive_md5_verified=True, archived_flacs_visited=files, selected=len(results), limit=args.limit,
        elapsed_s=perf_counter() - t0, keys_sha256=hashlib.sha256(asv.KEYS.read_bytes()).hexdigest()),
        indent=2) + "\n")
    for row in summary:
        print(row)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--asvspoof", action="store_true", help="Score ASVspoof with the models in --out")
    p.add_argument("--limit", type=int, default=0, help="Stop after this many archive FLACs (smoke test)")
    args = p.parse_args()
    args.out = args.out.resolve()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if args.asvspoof:
        if (args.out / "asvspoof_summary.csv").exists():
            p.error("ASVspoof results already exist in --out")
        asvspoof_stage(args, device)
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        dev_stage(args, device)


if __name__ == "__main__":
    main()
