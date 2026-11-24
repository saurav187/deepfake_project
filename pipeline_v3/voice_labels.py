"""Owner-run open-set speaker inference. Resume committed embedding batches on Drive."""

import argparse
import csv
import json
import sqlite3
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

from .data import digest
from .phonemedf import REFS, SYSTEMS, archive_name, member_id, write_csv


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def classify(views, references):
    scores = views @ references.T
    order = scores.argsort(axis=1)
    best = order[:, -1]
    floor = scores.max(axis=1).min()
    margin = np.min(
        scores.max(axis=1) - np.take_along_axis(scores, order[:, -2:-1], axis=1).ravel()
    )
    voice = (
        REFS[best[0]]
        if all(best == best[0]) and floor >= 0.86 and margin >= 0.08
        else "unknown"
    )
    return voice, float(floor), float(margin)


def run(data, batch):
    import io

    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from sklearn.cluster import MiniBatchKMeans
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    from .audio import SR

    root = data / "development"
    work = data / "stage0"
    model_dir = data / "models" / "wavlm-sv"
    model = WavLMForXVector.from_pretrained(model_dir, local_files_only=True).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    proc = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, local_files_only=True)
    db = sqlite3.connect(work / "sv.sqlite")
    db.execute(
        "CREATE TABLE IF NOT EXISTS vectors (g TEXT, uid TEXT, v BLOB, PRIMARY KEY(g,uid))"
    )
    db.execute("CREATE TABLE IF NOT EXISTS settings (signature TEXT PRIMARY KEY)")
    signature = json.dumps(
        [
            digest(work / "inventory.csv"),
            (model_dir / "revision.txt").read_text(),
            "views3-v1",
        ]
    )
    if db.execute("SELECT signature FROM settings").fetchall() not in (
        [],
        [(signature,)],
    ):
        raise ValueError("Voice cache belongs to another inventory or model")
    db.execute("INSERT OR IGNORE INTO settings VALUES (?)", (signature,))
    with (work / "inventory.csv").open() as f:
        totals = Counter(r["group"] for r in csv.DictReader(f))
    for g in ["references", *SYSTEMS]:
        done = {r[0] for r in db.execute("SELECT uid FROM vectors WHERE g=?", (g,))}
        if len(done) == totals[g]:
            continue
        pending = []
        processed = len(done)

        def flush(pending=pending, g=g):
            if not pending:
                return
            x = [v for _, views in pending for v in views]
            inputs = proc(
                x,
                sampling_rate=SR,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                emb = unit(model(**inputs).embeddings.float().cpu().numpy()).reshape(
                    len(pending), 2, -1
                )
            db.executemany(
                "INSERT INTO vectors VALUES (?,?,?)",
                [
                    (g, uid, v.astype("<f4").tobytes())
                    for (uid, _), v in zip(pending, emb, strict=True)
                ],
            )
            pending.clear()

        with tarfile.open(root / archive_name(g), "r|gz") as tar:
            for m in tar:
                if not m.isfile() or not m.name.lower().endswith(".wav"):
                    continue
                uid = (
                    Path(m.name).stem.split("_")[0]
                    if g == "references"
                    else member_id(m.name)
                )
                if uid in done:
                    continue
                x, sr = sf.read(
                    io.BytesIO(tar.extractfile(m).read()),
                    dtype="float32",
                    always_2d=True,
                )
                x = x.mean(axis=1)
                if sr != SR:
                    x = resample_poly(x, SR, sr)
                # Two non-overlapping views. Short clips are explicitly unknown.
                length = min(3 * SR, len(x) // 2)
                if length < SR:
                    db.execute("INSERT INTO vectors VALUES (?,?,?)", (g, uid, b""))
                    continue
                pending.append((uid, [x[:length], x[-length:]]))
                if len(pending) >= batch:
                    n = len(pending)
                    flush()
                    processed += n
                    if processed // 256 != (processed - n) // 256:
                        db.commit()
                        print(g, processed, totals[g], flush=True)
            flush()
            db.commit()

    def vectors(g):
        for uid, b in db.execute(
            "SELECT uid,v FROM vectors WHERE g=? ORDER BY uid", (g,)
        ):
            yield uid, np.frombuffer(b, "<f4").reshape(2, -1)

    refs = dict(vectors("references"))
    if set(refs) != set(REFS) or any(not v.size for v in refs.values()):
        raise ValueError("Missing usable reference embedding")
    anchors = unit(np.stack([refs[k].mean(0) for k in REFS]))
    out = []
    # Cluster only rejected embeddings. Clusters are descriptive; every accepted
    # built-in shares one train-only block even if the clustering fragments it.
    for g in SYSTEMS:
        rejected = []
        for uid, v in vectors(g):
            voice, score, margin = classify(v, anchors) if v.size else ("unknown", 0, 0)
            # Melo uses built-in voices. A strong resemblance to a VCTK speaker
            # is ambiguity, not evidence that the publisher used that reference.
            if g == "MeloTTS":
                voice = "unknown"
            out.append(
                {
                    "group": g,
                    "uid": uid,
                    "voice": voice,
                    "score": score,
                    "margin": margin,
                    "cluster": "-",
                }
            )
            if (
                v.size
                and voice == "unknown"
                and (v @ anchors.T).max() < 0.65
                and v[0] @ v[1] >= 0.80
            ):
                rejected.append((len(out) - 1, unit(v.mean(0))))
        if len(rejected) >= 50:
            X = np.stack([v for _, v in rejected])
            km = MiniBatchKMeans(
                n_clusters=min(10, len(X) // 25), random_state=41, n_init=3
            ).fit(X)
            centers = unit(km.cluster_centers_)
            freq = Counter(km.labels_)
            for (idx, v), label in zip(rejected, km.labels_, strict=True):
                out[idx]["cluster"] = f"{g}-{label}"
                if g == "MeloTTS" and freq[label] >= 25 and v @ centers[label] >= 0.88:
                    out[idx]["voice"] = "builtin-cohort"
    with (work / "inventory.csv").open() as f:
        expected = {
            (r["group"], r["uid"]) for r in csv.DictReader(f) if r["group"] in SYSTEMS
        }
    if {(r["group"], r["uid"]) for r in out} != expected:
        raise ValueError("Incomplete voice inference")
    db.close()
    write_csv(work / "voices.csv", out)
    (work / "voice-inventory.sha256").write_text(digest(work / "inventory.csv"))
    print(Counter((r["group"], r["voice"]) for r in out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    run(a.data, a.batch)
