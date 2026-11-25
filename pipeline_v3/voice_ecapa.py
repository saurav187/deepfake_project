"""ECAPA (SpeechBrain VoxCeleb) open-set voice inference, kept separate from WavLM's.

Owner-authorized 25 Nov 2026 after WavLM-SV collapsed (ten references nearly inseparable).
Products live in DATA/stage0-ecapa/ (own SQLite checkpoint, voices.csv, reports); the
WavLM cache in DATA/stage0/ is never opened for writing. Same confidence rule as
voice_labels.classify (best >= 0.86 on both views, same argmax, margin >= 0.08), same
voice-disjoint fold map. Views: two non-overlapping <=3 s windows of the speech-active
span (energy-gated trim, audio.prepare); clips whose active span is < 2 s are unknown.

  # model fetch (once):   python -m pipeline_v3.voice_ecapa fetch --data D
  # reference check:      python -m pipeline_v3.voice_ecapa refs --data D --work W
  # sample (cached):      python -m pipeline_v3.voice_ecapa embed --data D --work W --sample 16
  # full:                 python -m pipeline_v3.voice_ecapa embed --data D --work W
  # labels + report:      python -m pipeline_v3.voice_ecapa label --data D --work W [--sample 16]

W (the working stage directory, default DATA/stage0-ecapa) may be a local copy whose
inventory.csv is byte-identical to DATA/stage0/inventory.csv.
"""

import argparse
import csv
import hashlib
import io
import json
import shutil
import sqlite3
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .data import digest
from .phonemedf import FOLDS, REFS, SYSTEMS, VOICE_FOLD, archive_name, assign, member_id, write_csv
from .voice_labels import classify, unit

REPO = "speechbrain/spkrec-ecapa-voxceleb"
TAG = "ecapa-active-views-v1"


def fetch(data):
    from huggingface_hub import HfApi, snapshot_download

    dest = data / "models" / "ecapa-voxceleb"
    rev = HfApi().model_info(REPO).sha
    local = Path("/content/models/ecapa-voxceleb")
    snapshot_download(REPO, revision=rev, local_dir=local)
    (local / "revision.txt").write_text(rev)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(local, dest, ignore=shutil.ignore_patterns(".cache"))
    print(dest, rev)


def load_model(model_dir):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local = Path("/content/models/ecapa-run")
    if not local.exists():
        shutil.copytree(model_dir, local, ignore=shutil.ignore_patterns(".cache"))
    model = EncoderClassifier.from_hparams(
        source=str(local), savedir=str(local), run_opts={"device": device}
    )
    model.eval()
    return model, device


def views(x, sr):
    from scipy.signal import resample_poly

    from .audio import SR, prepare

    x = x.mean(axis=1)
    if sr != SR:
        x = resample_poly(x, SR, sr)
    t, _ = prepare(x.astype(np.float32))
    length = min(3 * SR, len(t) // 2)
    if length < SR:
        return None
    return [t[:length], t[-length:]]


def embed_batch(model, device, batch):
    import torch

    flat = [v for v in batch for v in v]
    n = max(len(v) for v in flat)
    wavs = torch.zeros(len(flat), n)
    for i, v in enumerate(flat):
        wavs[i, : len(v)] = torch.from_numpy(np.ascontiguousarray(v))
    lens = torch.tensor([len(v) / n for v in flat])
    with torch.inference_mode():
        e = model.encode_batch(wavs.to(device), lens.to(device)).squeeze(1).float().cpu().numpy()
    return unit(e).reshape(len(batch), 2, -1)


def sampled(uid, k):
    return k <= 1 or int(hashlib.sha256(("ecapa-sample/" + uid).encode()).hexdigest(), 16) % k == 0


def open_db(data, work):
    work.mkdir(parents=True, exist_ok=True)
    model_dir = data / "models" / "ecapa-voxceleb"
    db = sqlite3.connect(work / "sv-ecapa.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS vectors (g TEXT, uid TEXT, v BLOB, PRIMARY KEY(g,uid))")
    db.execute("CREATE TABLE IF NOT EXISTS settings (signature TEXT PRIMARY KEY)")
    sig = json.dumps([digest(work / "inventory.csv"), (model_dir / "revision.txt").read_text(), TAG])
    if db.execute("SELECT signature FROM settings").fetchall() not in ([], [(sig,)]):
        raise ValueError("ECAPA cache belongs to another inventory or model")
    db.execute("INSERT OR IGNORE INTO settings VALUES (?)", (sig,))
    db.commit()
    return db, model_dir


def embed(data, work, groups, k, batch):
    import soundfile as sf

    root = data / "development"
    db, model_dir = open_db(data, work)
    model, device = load_model(model_dir)
    with (work / "inventory.csv").open() as f:
        totals = Counter(r["group"] for r in csv.DictReader(f))
    for g in groups:
        done = {r[0] for r in db.execute("SELECT uid FROM vectors WHERE g=?", (g,))}
        if len(done) == totals[g]:
            continue
        pending, n, t0 = [], 0, time.time()

        def flush():
            nonlocal n
            if pending:
                emb = embed_batch(model, device, [v for _, v in pending])
                db.executemany(
                    "INSERT INTO vectors VALUES (?,?,?)",
                    [(g, u, e.astype("<f4").tobytes()) for (u, _), e in zip(pending, emb, strict=True)],
                )
                n += len(pending)
                pending.clear()

        with tarfile.open(root / archive_name(g), "r|gz") as tar:
            for m in tar:
                if not m.isfile() or not m.name.lower().endswith(".wav"):
                    continue
                uid = Path(m.name).stem.split("_")[0] if g == "references" else member_id(m.name)
                if uid in done or (g != "references" and not sampled(uid, k)):
                    continue
                x, sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32", always_2d=True)
                v = views(x, sr)
                if v is None:
                    db.execute("INSERT INTO vectors VALUES (?,?,?)", (g, uid, b""))
                    n += 1
                    continue
                pending.append((uid, v))
                if len(pending) >= batch:
                    before = n
                    flush()
                    if n // 512 != before // 512:
                        db.commit()
                        print(g, n, f"{n / (time.time() - t0):.1f} clips/s", flush=True)
            flush()
            db.commit()
        print(g, "done", n, f"{n / max(time.time() - t0, 1e-6):.1f} clips/s end-to-end", flush=True)
    db.close()


def vectors(db, g):
    for uid, b in db.execute("SELECT uid,v FROM vectors WHERE g=? ORDER BY uid", (g,)):
        yield uid, (np.frombuffer(b, "<f4").reshape(2, -1) if b else None)


def anchors(db):
    refs = dict(vectors(db, "references"))
    if set(refs) != set(REFS) or any(v is None for v in refs.values()):
        raise ValueError("Missing usable reference embedding")
    return refs, unit(np.stack([refs[k].mean(0) for k in REFS]))


def refs_report(data, work):
    db, _ = open_db(data, work)
    if db.execute("SELECT count(*) FROM vectors WHERE g='references'").fetchone()[0] < len(REFS):
        db.close()
        embed(data, work, ["references"], 1, 10)
        db, _ = open_db(data, work)
    refs, A = anchors(db)
    M = A @ A.T
    off = M[~np.eye(len(REFS), dtype=bool)]
    same = [float(unit(refs[k][0]) @ unit(refs[k][1])) for k in REFS]
    report = {
        "model": REPO,
        "cross_reference_cosine": {"min": float(off.min()), "median": float(np.median(off)), "max": float(off.max())},
        "same_reference_two_views": dict(zip(REFS, np.round(same, 3).tolist())),
        "closest_pairs": sorted(
            ((REFS[i], REFS[j], round(float(M[i, j]), 3)) for i in range(10) for j in range(i + 1, 10)),
            key=lambda t: -t[2],
        )[:5],
    }
    (work / "refs-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def label(data, work, k):
    db, _ = open_db(data, work)
    _, A = anchors(db)
    with (work / "inventory.csv").open() as f:
        inv = [r for r in csv.DictReader(f) if r["group"] in SYSTEMS]
    reader = {(r["group"], r["uid"]): r["reader"] for r in inv}
    out, report = [], {"sample_every": k, "systems": {}}
    from sklearn.cluster import MiniBatchKMeans

    for g in SYSTEMS:
        rejected, start = [], len(out)
        for uid, v in vectors(db, g):
            voice, score, margin = classify(v, A) if v is not None else ("unknown", 0, 0)
            if g == "MeloTTS":
                voice = "unknown"
            out.append({"group": g, "uid": uid, "voice": voice, "score": score, "margin": margin, "cluster": "-"})
            if v is not None and voice == "unknown" and (v @ A.T).max() < 0.65 and v[0] @ v[1] >= 0.80:
                rejected.append((len(out) - 1, unit(v.mean(0))))
        if len(rejected) >= 50:
            X = np.stack([v for _, v in rejected])
            km = MiniBatchKMeans(n_clusters=min(10, len(X) // 25), random_state=41, n_init=3).fit(X)
            centers = unit(km.cluster_centers_)
            freq = Counter(km.labels_)
            for (idx, v), lab in zip(rejected, km.labels_, strict=True):
                out[idx]["cluster"] = f"{g}-{lab}"
                if g == "MeloTTS" and freq[lab] >= 25 and v @ centers[lab] >= 0.88:
                    out[idx]["voice"] = "builtin-cohort"
        rows = out[start:]
        if not rows:
            continue
        folds = defaultdict(lambda: {"clips": 0, "readers": set(), "voices": set()})
        for r in rows:
            split = assign("librispeech/" + reader[g, r["uid"]])
            target = "train" if r["voice"] == "builtin-cohort" else VOICE_FOLD.get(r["voice"])
            if target == split and not (g == "ChatterboxTTS" and split == "train"):
                folds[split]["clips"] += 1
                folds[split]["readers"].add(reader[g, r["uid"]])
                folds[split]["voices"].add(r["voice"])
        sc = np.array([r["score"] for r in rows])
        mg = np.array([r["margin"] for r in rows])
        report["systems"][g] = {
            "embedded": len(rows),
            "voices": dict(Counter(r["voice"] for r in rows).most_common()),
            "best_score_pct_5_25_50_75_95": np.percentile(sc, [5, 25, 50, 75, 95]).round(3).tolist(),
            "margin_pct_5_25_50_75_95": np.percentile(mg, [5, 25, 50, 75, 95]).round(3).tolist(),
            "retained_by_fold": {
                f: {"clips": folds[f]["clips"], "projected_full": folds[f]["clips"] * k,
                    "readers": len(folds[f]["readers"]), "voices": sorted(folds[f]["voices"])}
                for f in FOLDS
            },
        }
    name = "sample" if k > 1 else "full"
    (work / f"label-report-{name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if k <= 1:
        expected = {(r["group"], r["uid"]) for r in inv}
        if {(r["group"], r["uid"]) for r in out} != expected:
            raise ValueError("Incomplete ECAPA voice inference")
        write_csv(work / "voices.csv", out)
        (work / "voice-inventory.sha256").write_text(digest(work / "inventory.csv"))
    db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["fetch", "refs", "embed", "label"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--work", type=Path)
    p.add_argument("--sample", type=int, default=1, help="embed/label every k-th utterance by hash")
    p.add_argument("--groups", nargs="+", default=["references", "XTTSv2", "ChatterboxTTS", "MeloTTS"])
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()
    work = a.work or a.data / "stage0-ecapa"
    if a.action == "fetch":
        fetch(a.data)
        return
    if not (work / "inventory.csv").exists():
        work.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.data / "stage0" / "inventory.csv", work / "inventory.csv")
    if digest(work / "inventory.csv") != digest(a.data / "stage0" / "inventory.csv"):
        raise ValueError("Working inventory differs from stage0/inventory.csv")
    if a.action == "refs":
        refs_report(a.data, work)
    elif a.action == "embed":
        embed(a.data, work, a.groups, a.sample, a.batch)
    else:
        label(a.data, work, a.sample)


if __name__ == "__main__":
    main()
