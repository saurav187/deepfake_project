"""Compare native-format training to a size-, speaker- and label-matched control."""
import argparse
from collections import Counter, defaultdict
import csv
import importlib
import json
import os
from pathlib import Path
import random
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from audio_features import BASE, ROOT, RUN
from prepare_manifest import write_csv

baseline = importlib.import_module("02_train_cnn_lstm")
SEEDS = (42, 43, 44)
EPOCHS = 5
BUDGET = 0.05


def select_cohorts(rows):
    files = baseline.group_files([r for r in rows if r["split"] in ("train", "dev")])
    pools = defaultdict(list)
    formats = {}
    train = {"native": [], "mixed": []}
    dev = []
    for name, chunks in sorted(files.items()):
        info = sf.info(ROOT / name)
        formats[name] = (info.samplerate, info.channels)
        row = chunks[0]
        if row["split"] == "train":
            pools[row["speaker_group"], row["label"]].append(name)
        elif formats[name] == (16000, 1):
            dev.extend(chunks)

    rng = random.Random(2026)
    membership = []
    for (speaker, label), pool in sorted(pools.items()):
        native = [name for name in pool if formats[name] == (16000, 1)]
        mixed = rng.sample(pool, len(native))
        # Shared recordings retain identical input. Pair the remaining files by
        # length within each speaker/class and give the longer one the same budget.
        shared = sorted(set(native) & set(mixed))
        sort_key = lambda name: (len(files[name]), name)
        pairs = [(name, name) for name in shared]
        pairs.extend(zip(sorted(set(native) - set(shared), key=sort_key),
                         sorted(set(mixed) - set(shared), key=sort_key), strict=True))
        for a, b in pairs:
            count = min(len(files[a]), len(files[b]))
            for arm, name in (("native", a), ("mixed", b)):
                chunks = sorted(files[name], key=lambda r: int(r["chunk"]))
                idx = np.linspace(0, len(chunks)-1, count, dtype=int) if count > 1 else [len(chunks)//2]
                selected = [chunks[i] for i in idx]
                train[arm].extend(selected)
                membership.append(dict(arm=arm, source_file=name, paired_file=b if arm=="native" else a,
                                       speaker=speaker, label=label, rate=formats[name][0], channels=formats[name][1],
                                       available_chunks=len(chunks), row_ids=" ".join(r["row_id"] for r in selected)))
    signature = lambda chunks: Counter((r["speaker_group"], r["label"]) for r in chunks)
    assert signature(train["native"]) == signature(train["mixed"])
    assert len(dev) and len({r["source_file"] for r in dev}) == 734
    for chunks in train.values():
        assert len({r["source_file"] for r in chunks}) == 2335
        assert len({r["row_id"] for r in chunks}) == len(chunks)
    return train, dev, membership


def fit(dataset, rows, dev_loader, dev, device, seed, out):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=96, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = baseline.CNNLSTMDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.0001)
    counts = Counter(r["label"] for r in rows)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([counts["0"]/counts["1"]], device=device))
    epochs = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for epoch in range(1, EPOCHS + 1):
        start = perf_counter()
        model.train()
        seen = []
        for x, y, ids in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=="cuda"):
                loss = loss_fn(model(x.to(device)).flatten(), y.to(device))
            loss.backward()
            optimizer.step()
            seen.extend(ids.tolist())
        baseline.verify_ids(seen, rows)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epochs.append(perf_counter() - start)
        print(f"{out.name} epoch {epoch}/{EPOCHS}: {epochs[-1]:.2f}s", flush=True)
    torch.save(model.state_dict(), out / "checkpoint.pth")
    scores = baseline.predict(model, dev_loader, dev, device)
    return baseline.records(dev, scores, {"aggregation":"mean", "threshold":.5}), epochs


def measure(items, arm, seed):
    genuine = sorted((r["score"] for r in items if not r["label"]), reverse=True)
    # The dev ROC operating point is descriptive. No deployment rule is fitted.
    # Exclude the next highest genuine score, giving at most floor(0.05*n) errors.
    cutoff = float(np.nextafter(genuine[int(BUDGET * len(genuine))], np.inf))
    metrics = []
    for speaker in ["all", *sorted({r["speaker_group"] for r in items})]:
        group = [r for r in items if speaker=="all" or r["speaker_group"]==speaker]
        y = np.array([r["label"] for r in group])
        s = np.array([r["score"] for r in group])
        pred = s >= cutoff
        real, fake = int(sum(y==0)), int(sum(y==1))
        fp, tp = int(sum(pred & (y==0))), int(sum(pred & (y==1)))
        metrics.append(dict(arm=arm, seed=seed, speaker=speaker, auc=roc_auc_score(y,s) if real and fake else None,
                            cutoff=cutoff, real=real, fake=fake, fp=fp, tp=tp,
                            fpr=fp/real if real else None, recall=tp/fake if fake else None))
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    args = p.parse_args()
    experiment_start = perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    manifest = RUN / "manifest.csv"
    train, dev, membership = select_cohorts(baseline.read_manifest(manifest))
    write_csv(args.out / "membership.csv", membership)
    config = dict(manifest_sha256=baseline.sha(manifest), script_sha256=baseline.sha(Path(__file__)),
                  seeds=SEEDS, sample_seed=2026, epochs=EPOCHS, fpr_budget=BUDGET, batch_size=96,
                  torch=str(torch.__version__), cuda=torch.version.cuda, device=str(device),
                  gpu=torch.cuda.get_device_name() if device.type=="cuda" else None,
                  train_chunks_per_arm=len(train["native"]), dev_chunks=len(dev))
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    prep_start = perf_counter()
    dev_loader = baseline.make_loader(dev, False, 42)
    prep = {"dev": perf_counter() - prep_start}
    metrics, predictions, timings = [], [], []
    for arm, rows in train.items():
        prep_start = perf_counter()
        data = baseline.make_loader(rows, False, 42).dataset
        prep[arm] = perf_counter() - prep_start
        for seed in SEEDS:
            out = args.out / f"{arm}_{seed}"
            out.mkdir()
            fit_start = perf_counter()
            items, epochs = fit(data, rows, dev_loader, dev, device, seed, out)
            timings.append(dict(arm=arm, seed=seed, training_s=sum(epochs), fit_and_eval_s=perf_counter()-fit_start,
                                **{f"epoch_{i}_s": t for i,t in enumerate(epochs, 1)}))
            metrics.extend(measure(items, arm, seed))
            predictions.extend(dict(arm=arm, seed=seed, file=r["source_file"], speaker=r["speaker_group"],
                                    label=r["label"], score=r["score"]) for r in items)
            print(f"Finished {arm}, seed {seed}", flush=True)
        del data
    path = BASE / "research/asvspoof_la_v1/matched_format_dev.csv"
    with path.open() as f:
        reference = [dict(source_file=r["file"], speaker_group=r["speaker"], label=int(r["label"]),
                          score=float(r["score"])) for r in csv.DictReader(f)]
    assert {(r["source_file"], r["label"], r["speaker_group"]) for r in reference} == {
        (r["source_file"], int(r["label"]), r["speaker_group"]) for r in dev}
    metrics.extend(measure(reference, "historical_full_train", 42))
    write_csv(args.out / "dev_predictions.csv", predictions)
    write_csv(args.out / "metrics.csv", metrics)
    write_csv(args.out / "timing.csv", timings)
    (args.out / "elapsed.json").write_text(json.dumps(dict(feature_preparation_s=prep,
        total_s=perf_counter()-experiment_start), indent=2))


if __name__ == "__main__":
    main()
