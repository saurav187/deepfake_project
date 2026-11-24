"""Cache pooled SSL states once per clip/channel/encoder and fit recording-level probes."""

import hashlib
import json
import os
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

from .audio import cepstra, channel, decode, model_view, windows
from .data import digest, entries, payload
from .metrics import bootstrap, metrics, slices


def cache(cfg, rows, encoder, tag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        AutoModel.from_pretrained(str(encoder), local_files_only=True).to(device).eval()
    )
    signature = hashlib.sha256(
        (
            (
                (encoder / "revision.txt").read_text()
                if (encoder / "revision.txt").exists()
                else str(encoder.resolve())
            )
            + digest(encoder / "config.json")
            + digest(cfg.local / "index.json")
            + "v3-pool-3"
        ).encode()
    ).hexdigest()[:16]
    directory = cfg.data / "features" / tag / signature
    directory.mkdir(parents=True, exist_ok=True)
    features = {}
    with torch.inference_mode():
        for r in rows:
            for view in ("clean", "opus"):
                key = r["id"], view
                path = directory / f"{r['id']}-{view}.npz"
                if not path.exists():
                    x = model_view(channel(decode(payload(cfg.local, r)), view), r)
                    pooled = []
                    lfcc = []
                    for _, w in windows(x):
                        h = model(
                            torch.from_numpy(w).to(device)[None],
                            output_hidden_states=True,
                        ).hidden_states
                        pooled.append(
                            torch.stack(
                                [
                                    torch.cat(
                                        [v[0].mean(0), v[0].std(0, unbiased=False)]
                                    )
                                    for v in h
                                ]
                            )
                            .cpu()
                            .numpy()
                        )
                        lfcc.append(cepstra(w))
                    tmp = path.with_suffix(".tmp")
                    with tmp.open("wb") as f:
                        np.savez(
                            f,
                            h=np.mean(pooled, axis=0).astype("float16")
                            if pooled
                            else [],
                            lf=np.mean(lfcc, axis=0) if lfcc else [],
                        )
                    os.replace(tmp, path)
                with np.load(path) as a:
                    if a["h"].size:
                        features[key] = (a["h"], a["lf"])
    return features


def pilot(cfg):
    rng = np.random.default_rng(cfg.seed)
    allrows = entries(cfg.local, {"train", "select"})
    # Replay the family holdout on a final pack that also contains training Gemini.
    # This restriction applies to feature probes only. The teacher uses the full pack.
    holdout = {"gemini", "freevc24", "chatterboxtts"}
    allrows = [
        dict(r, domain="unseen") if r["generator"] in holdout else r
        for r in allrows
        if not (r["split"] == "train" and r["generator"] in holdout)
    ]
    rows = []
    for split in ("train", "select"):
        for label in ("0", "1"):
            pool = [r for r in allrows if r["split"] == split and r["label"] == label]
            limit = cfg.per_class if split == "train" else cfg.per_class // 2
            if split == "select":
                # Sample real parents and their fakes together for language-specific EER.
                if label == "0":
                    continue
                for lang in ("en", "hi"):
                    controls = {
                        r["parent"]: r
                        for r in allrows
                        if r["language"] == lang
                        and r["split"] == split
                        and r["label"] == "0"
                    }
                    chosen = {}
                    for domain in ("unseen", "seen"):
                        candidates = [
                            r
                            for r in pool
                            if r["language"] == lang
                            and r["domain"] == domain
                            and r["parent"] in controls
                        ]
                        added = 0
                        for i in rng.permutation(len(candidates)):
                            r = candidates[i]
                            if r["parent"] not in chosen:
                                chosen[r["parent"]] = r
                                added += 1
                                if added >= max(1, limit // 4):
                                    break
                    for parent, fake in chosen.items():
                        rows.extend((controls[parent], fake))
            else:
                first = len(rows)
                for lang in ("en", "hi"):
                    group = [r for r in pool if r["language"] == lang]
                    rows.extend(
                        group[i] for i in rng.permutation(len(group))[: limit // 2]
                    )
                used = {r["id"] for r in rows[first:]}
                rest = [r for r in pool if r["id"] not in used]
                rows.extend(
                    rest[i] for i in rng.permutation(len(rest))[: limit - len(used)]
                )
    report = {
        "probe_holdout": sorted(holdout),
        "selected": dict(Counter(r["split"] for r in rows)),
        "models": {},
    }
    for encoder, tag in [(cfg.encoder, "xlsr"), (cfg.small_encoder, "small")]:
        data = cache(cfg, rows, encoder, tag)
        if not data:
            raise ValueError("No usable pilot clips")
        n = next(iter(data.values()))[0].shape[0]
        specs = {f"layer-{i}": (i, i + 1) for i in range(n)}
        specs.update(first4=(1, 5), first6=(1, 7), full=(1, n))
        if tag == "xlsr":
            specs["lfcc"] = None
        for rep, bounds in specs.items():
            for method in (
                ("linear", "mlp")
                if rep in {"first4", "first6", "full", "lfcc"}
                else ("linear",)
            ):

                def features(r, view, data=data, bounds=bounds):
                    h, lf = data[r["id"], view]
                    return (
                        lf
                        if bounds is None
                        else h[slice(*bounds)].mean(0, dtype=np.float32)
                    )

                train = [
                    r
                    for r in rows
                    if r["split"] == "train" and (r["id"], "clean") in data
                ]
                estimator = (
                    LogisticRegression(max_iter=400, class_weight="balanced")
                    if method == "linear"
                    else MLPClassifier(
                        hidden_layer_sizes=(64,), max_iter=80, random_state=cfg.seed
                    )
                )
                m = make_pipeline(StandardScaler(), estimator)
                m.fit(
                    [features(r, "clean") for r in train],
                    [int(r["label"]) for r in train],
                )
                for view in ("clean", "opus"):
                    dev = [
                        r
                        for r in rows
                        if r["split"] == "select"
                        and all((r["id"], v) in data for v in ("clean", "opus"))
                    ]
                    scores = m.predict_proba([features(r, view) for r in dev])[:, 1]
                    stats = metrics([int(r["label"]) for r in dev], scores)
                    stats.update(
                        slices=slices(dev, scores),
                        eligible=len(dev),
                        selected=report["selected"]["select"],
                    )
                    if rep in {"full", "lfcc", "first4", "first6"}:
                        stats.update(bootstrap(dev, scores, 200))
                    report["models"][f"{tag}/{rep}/{method}/{view}"] = stats
    selected = report["models"]
    baseline = (
        selected["xlsr/lfcc/linear/clean"]["slices"].get("domain/unseen", {}).get("eer")
    )
    decisions = {}
    for name, result in selected.items():
        if not name.endswith("/clean") or "/lfcc/" in name or "/layer-" in name:
            continue
        opus = selected[name[:-5] + "opus"]
        e = result["slices"].get("domain/unseen", {}).get("eer")
        o = opus["slices"].get("domain/unseen", {}).get("eer")
        coverage = result["eligible"] / result["selected"]
        languages = [
            result["slices"].get("language/" + lang + "/unseen", {}).get("eer")
            for lang in ("hi", "en")
        ]
        decisions[name] = (
            min(
                sum(r["split"] == "train" and r["label"] == lab for r in rows)
                for lab in ("0", "1")
            )
            >= 1000
            and None not in (baseline, e, o, *languages)
            and e <= 0.20
            and baseline - e >= 0.03
            and o - e <= 0.05
            and max(languages) <= 0.25
            and coverage >= 0.95
        )
    report.update(
        index_sha256=digest(cfg.local / "index.json"),
        criteria=decisions,
        go=any(decisions.values()),
    )
    out = cfg.data / "pilot.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(out)
    print(out)
