"""ECAPA (SpeechBrain VoxCeleb) open-set voice inference, kept separate from WavLM's.

Owner-authorized 25 Nov 2026 after WavLM-SV collapsed (ten references nearly inseparable).
Products live in DATA/stage0-ecapa/ (own SQLite checkpoint, voices.csv, reports); the
WavLM cache in DATA/stage0/ is never opened for writing. Same confidence rule as
rule shape as voice_labels.classify (same argmax, margin >= 0.08) with an ECAPA floor frozen by `calibrate` (rule.json), same
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
from .voice_labels import unit

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


def classify_rule(views, references, floor, margin):
    """Frozen ECAPA rule: both views choose one voice, min best >= floor, min margin >= margin."""
    scores = views @ references.T
    order = np.sort(scores, axis=1)
    best = scores.argmax(axis=1)
    lo, mg = float(order[:, -1].min()), float((order[:, -1] - order[:, -2]).min())
    ok = best[0] == best[1] and lo >= floor and mg >= margin
    return (REFS[best[0]] if ok else "unknown"), lo, mg


def fold_table(rows, reader):
    folds = defaultdict(lambda: {"clips": 0, "readers": set(), "voices": set()})
    for r in rows:
        g = r["group"]
        split = assign("librispeech/" + reader[g, r["uid"]])
        target = "train" if r["voice"] == "builtin-cohort" else VOICE_FOLD.get(r["voice"])
        if target == split and not (g == "ChatterboxTTS" and split == "train"):
            folds[split]["clips"] += 1
            folds[split]["readers"].add(reader[g, r["uid"]])
            folds[split]["voices"].add(r["voice"])
    return {
        f: {"pairs": folds[f]["clips"], "readers": len(folds[f]["readers"]), "voices": sorted(folds[f]["voices"])}
        for f in FOLDS
    }


def label(data, work, k):
    """Apply the frozen rule.json to XTTSv2/Chatterbox; MeloTTS rows come from WavLM's voices.csv."""
    rule = json.loads((work / "rule.json").read_text())
    if not rule.get("supported") or rule["rule"] != "ecapa-reference-floor-v1":
        raise ValueError("No supported frozen ECAPA rule")
    db, model_dir = open_db(data, work)
    if (model_dir / "revision.txt").read_text() != rule["revision"]:
        raise ValueError("Rule was calibrated on another model revision")
    _, A = anchors(db)
    with (work / "inventory.csv").open() as f:
        inv = [r for r in csv.DictReader(f) if r["group"] in SYSTEMS]
    reader = {(r["group"], r["uid"]): r["reader"] for r in inv}
    report = {"rule": {x: rule[x] for x in ("rule", "floor", "margin", "strict_floor_sensitivity_only")},
              "sample_every": k, "labels": "INFERRED", "systems": {}}
    out, strict = [], []
    for g in ("XTTSv2", "ChatterboxTTS"):
        rows, srows = [], []
        for uid, v in vectors(db, g):
            if v is None:
                voice, lo, mg, sv = "unknown", 0.0, 0.0, "unknown"
            else:
                voice, lo, mg = classify_rule(v, A, rule["floor"], rule["margin"])
                sv = classify_rule(v, A, rule["strict_floor_sensitivity_only"], rule["margin"])[0]
            rows.append({"group": g, "uid": uid, "voice": voice, "score": lo, "margin": mg, "cluster": "-"})
            srows.append({"group": g, "uid": uid, "voice": sv})
        out += rows
        strict += srows
        report["systems"][g] = {
            "embedded": len(rows),
            "accepted": sum(r["voice"] != "unknown" for r in rows),
            "acceptance_rate": sum(r["voice"] != "unknown" for r in rows) / max(1, len(rows)),
            "unknown": sum(r["voice"] == "unknown" for r in rows),
            "targets": dict(Counter(r["voice"] for r in rows if r["voice"] != "unknown").most_common()),
            "folds": fold_table(rows, reader),
            "strict_sensitivity": {
                "acceptance_rate": sum(r["voice"] != "unknown" for r in srows) / max(1, len(srows)),
                "folds": fold_table(srows, reader),
            },
        }
    melo = [r for r in csv.DictReader((data / "stage0" / "voices.csv").open()) if r["group"] == "MeloTTS"] \
        if (data / "stage0" / "voices.csv").exists() else []
    if melo:
        if (data / "stage0" / "voice-inventory.sha256").read_text().strip() != digest(work / "inventory.csv"):
            raise ValueError("WavLM Melo labels belong to another inventory")
        out += melo
        report["systems"]["MeloTTS"] = {
            "source": "WavLM stage0/voices.csv (built-in voices; train-only cohort)",
            "embedded": len(melo),
            "voices": dict(Counter(r["voice"] for r in melo)),
            "folds": fold_table(melo, reader),
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


MARGIN = 0.08
MIN_GENUINE_ACCEPT = 0.80  # fixed before any trial: below this the references cannot support a cutoff
STRICT_DELTA = 0.05  # sensitivity check only; never used to retune


def calibrate(data, work):
    """Known-identity reference trials -> frozen ECAPA rule (independent of corpus and folds).

    Each reference's speech-active span is cut into three non-overlapping thirds. In
    rotation k, third k enrolls every reference and the other two thirds are the clip's two
    query windows. Trials: correct reference vs nine wrong ones (closed set), and the same
    query with its correct reference removed (every available answer wrong; must reject).
    Floor = smallest 0.01 step above every wrong-reference window score seen in any trial,
    so no wrong reference can pass on score alone. The rule is frozen in rule.json.
    """
    import soundfile as sf

    from .audio import SR, prepare

    model, device = load_model(data / "models" / "ecapa-voxceleb")
    audio = {}
    with tarfile.open(data / "development" / archive_name("references"), "r|gz") as tar:
        for m in tar:
            if m.isfile() and m.name.lower().endswith(".wav"):
                x, sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32", always_2d=True)
                x = x.mean(axis=1)
                if sr != SR:
                    from scipy.signal import resample_poly

                    x = resample_poly(x, SR, sr).astype(np.float32)
                audio[Path(m.name).stem.split("_")[0]] = prepare(x)[0]
    if set(audio) != set(REFS):
        raise ValueError("Missing reference recordings")
    thirds = {}
    for k in REFS:
        n = len(audio[k]) // 3
        thirds[k] = [audio[k][i * n : (i + 1) * n] for i in range(3)]
    emb = {k: embed_batch(model, device, [[t[0], t[1]], [t[2], t[2]]]) for k, t in thirds.items()}
    seg = {k: [emb[k][0][0], emb[k][0][1], emb[k][1][0]] for k in REFS}
    trials, wrong_scores = [], []
    for rot in range(3):
        anchors = np.stack([seg[k][rot] for k in REFS])
        for i, k in enumerate(REFS):
            q = np.stack([seg[k][j] for j in range(3) if j != rot])
            s = q @ anchors.T
            wrong = np.delete(s, i, axis=1)
            wrong_scores.extend(wrong.ravel().tolist())
            o = np.sort(s, axis=1)
            ow = np.sort(wrong, axis=1)
            trials.append({
                "reference": k, "rotation": rot,
                "views_choose_correct": bool((s.argmax(1) == i).all()),
                "correct_min_score": float(s[:, i].min()),
                "closed_margin_min": float((o[:, -1] - o[:, -2]).min()),
                "best_wrong_score": float(wrong.max()),
                "best_wrong_voice": REFS[[j for j in range(10) if j != i][int(wrong.max(0).argmax())]],
                "removed_agree": bool(wrong.argmax(1)[0] == wrong.argmax(1)[1]),
                "removed_margin_min": float((ow[:, -1] - ow[:, -2]).min()),
                "removed_best_min": float(wrong.max(1).min()),
            })
    floor = float(np.ceil((max(wrong_scores) + 1e-9) * 100) / 100)

    def accept_genuine(t, f):
        return t["views_choose_correct"] and t["closed_margin_min"] >= MARGIN and t["correct_min_score"] >= f

    def accept_removed(t, f):
        return t["removed_agree"] and t["removed_margin_min"] >= MARGIN and t["removed_best_min"] >= f

    def rates(f):
        return {
            "genuine_accept": sum(accept_genuine(t, f) for t in trials) / len(trials),
            "removed_false_accept": sum(accept_removed(t, f) for t in trials) / len(trials),
        }

    A = unit(np.stack([np.mean(seg[k], axis=0) for k in REFS]))
    M = A @ A.T
    np.fill_diagonal(M, -1)
    i, j = np.unravel_index(M.argmax(), M.shape)
    result = rates(floor)
    rule = {
        "rule": "ecapa-reference-floor-v1",
        "model": REPO,
        "revision": (data / "models" / "ecapa-voxceleb" / "revision.txt").read_text(),
        "views": "two non-overlapping <=3 s windows of the speech-active span; same argmax required",
        "margin": MARGIN,
        "floor": floor,
        "floor_definition": "smallest 0.01 step above every wrong-reference window score in the trials",
        "strict_floor_sensitivity_only": round(floor + STRICT_DELTA, 2),
        "trials": len(trials),
        "reference_active_seconds": {k: round(len(audio[k]) / SR, 2) for k in REFS},
        "window_seconds": {k: round(len(thirds[k][0]) / SR, 2) for k in REFS},
        "max_wrong_score": max(wrong_scores),
        "at_floor": result,
        "at_strict_floor": rates(floor + STRICT_DELTA),
        "hardest_pair": [REFS[i], REFS[j], round(float(M[i, j]), 3)],
        "per_reference": {
            k: {
                "genuine_accepted": sum(accept_genuine(t, floor) for t in trials if t["reference"] == k),
                "of": 3,
                "correct_min_score": round(min(t["correct_min_score"] for t in trials if t["reference"] == k), 3),
                "best_wrong_score": round(max(t["best_wrong_score"] for t in trials if t["reference"] == k), 3),
                "hardest_wrong": Counter(t["best_wrong_voice"] for t in trials if t["reference"] == k).most_common(1)[0][0],
                "removed_false_accepts": sum(accept_removed(t, floor) for t in trials if t["reference"] == k),
            }
            for k in REFS
        },
        "supported": result["genuine_accept"] >= MIN_GENUINE_ACCEPT and result["removed_false_accept"] == 0,
        "caveat": "Ten single recordings; same-recording matching is easier than matching a synthesized voice. Accepted assignments are INFERRED.",
        "trial_detail": trials,
    }
    (work / "rule.json").write_text(json.dumps(rule, indent=2))
    print(json.dumps({k: v for k, v in rule.items() if k != "trial_detail"}, indent=2))
    return rule


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["fetch", "refs", "calibrate", "embed", "label"])
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
    elif a.action == "calibrate":
        calibrate(a.data, work)
    elif a.action == "embed":
        embed(a.data, work, a.groups, a.sample, a.batch)
    else:
        label(a.data, work, a.sample)


if __name__ == "__main__":
    main()
