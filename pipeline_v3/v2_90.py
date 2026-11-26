"""V2-90: train/calibrate/evaluate a modern-TTS detector (protocol college/28_V2_90_PROTOCOL.md).

Phases (all outputs under --work, mirrored to --drive):
  prep      raw MLAAD tars + genuine parquet + M-AILABS tgz -> split-assigned 4 s crops (int16 npy shards)
  audit     simple-feature shortcut probes, train -> development (reported, non-blocking)
  train     fine-tune XLS-R-300m (weighted layers + stat pooling head), label-symmetric channel aug
  calibrate score calibration genuine, freeze cutoff (lowest cutoff with <= floor(.15 n) flagged)
  signoff   bind model sha, cutoff, test manifest sha (write-once) before any test score
  evaluate  score test once: clean + simulated call channels, per family / source / speaker
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import tarfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

SR, WIN = 16000, 64000
TARGETS = ["Chatterbox", "ElevenLabs-v3", "FishTTS", "Gemini-3.1-Flash-TTS"]
VOICE_SPLIT = ("ElevenLabs", "Gemini")  # stock-voice families: half A train, half B test
RELATED = {"Chatterbox": ["Chatterbox-Turbo", "Resemble.ai (April 12th, 2025)"],
           "ElevenLabs-v3": ["ElevenLabs-Turbo-v2.5", "ElevenLabs-v2-Multilingual"],
           "FishTTS": ["Fish-S2-Pro", "Openaudio-S1-Mini"], "Gemini-3.1-Flash-TTS": []}
GEN_CAP_CORPUS, GEN_CAP_SPK, EVAL_CAP_SPK = 4000, 40, 10
CHANNELS_EVAL = ["clean", "mulaw8k", "g726", "opus12", "gsm"]
TRAIN_CODECS = ["mulaw8k", "alaw8k", "gsm", "opus8", "opus16", "opus24", "mp3"]  # G.726 held out (unseen codec)
IST = timezone(timedelta(hours=5, minutes=30))
HINDI_SOURCES = ("kathbath", "indicvoices")
SHIPPED_RELEASE_CUTOFF = 0.5623497986447662  # detector/runs/release/v1/release.json likely_fake_at_or_above
# Documented minimum coverage (protocol 28 amendment 6). Fakes: eligible recordings per target family;
# genuine: (recordings, speakers) per split. Below these, training/selection is refused or a cell is insufficient.
MIN_FAKE = {"train": 200, "dev": 25, "test": 100}
MIN_GEN = {"train": (3000, 100), "dev": (200, 20), "cal": (500, 50), "test": (500, 60)}
PREPARED = "prepared"
HINDI_CAP_FAKE = 3000


def split_lang(folder):
    """MLAAD tar stem: '<lang>__<system>' for non-English folders, plain '<system>' for English."""
    return folder.split("__", 1) if "__" in folder else ("en", folder)


def base_family(fam):
    return split_lang(fam)[1]


def h(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def fsha(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def now():
    t = datetime.now(timezone.utc)
    return {"utc": t.isoformat(timespec="seconds"), "ist": t.astimezone(IST).isoformat(timespec="seconds")}


# ------------------------------------------------------------------------- preprocessing
def crop(raw):
    """Shared treatment (trim, centre 4 s, RMS norm) -> int16 array or None."""
    from .audio import decode
    from .clip_cache import treat

    b = treat(raw)
    if b is None:
        return None
    x = decode(b)
    if len(x) < WIN:
        return None
    return (np.clip(x[:WIN], -1, 32767 / 32768) * 32768).round().astype("<i2")


def mlaad_voice(row):
    v = (row.get("reference_speaker") or "").strip()
    return v if v and v != "-" else Path(row["path"]).stem.split("_")[0]


def mlaad_rows(raw):
    """Assign every MLAAD clip a split by protocol 28 section 2."""
    out = []
    for tp in sorted((raw / "mlaad").glob("*.tar")):
        folder = tp.stem
        with tarfile.open(tp) as t:
            try:
                meta = t.extractfile(f"{folder}/meta.csv").read().decode()
            except KeyError:  # meta lost on a restarted download: publisher meta.csv sidecar (same revision)
                side = raw / "mlaad-meta" / f"{folder}.csv"
                if not side.exists():
                    raise ValueError(f"{tp.name}: no meta.csv and no sidecar {side}")
                meta = side.read_text()
        lang, base = split_lang(folder)
        for r in csv.DictReader(io.StringIO(meta), delimiter="|"):
            region = r["original_file"].split("/")[0]
            voice = mlaad_voice(r)
            half = "A" if h(voice) % 2 == 0 else "B"
            stock = base.startswith(VOICE_SPLIT)
            if lang == "de":  # other-language training data of a target family (never test)
                if stock and half == "B":
                    continue
                split = "dev" if h(r["original_file"]) % 10 == 0 else "train"
            elif lang == "hi":  # Hindi: same M-AILABS text roles as English (en_US train, en_UK test)
                split = ("dev" if h(r["original_file"]) % 10 == 0 else "train") if region == "en_US" else "test"
            elif region == "en_US":
                if stock and half == "B":
                    continue
                split = "dev" if h(r["original_file"]) % 10 == 0 else "train"
            else:  # en_UK: test for the four targets, never training for anyone
                if folder not in TARGETS or (stock and half == "A"):
                    continue
                split = "test"
            out.append(dict(uid=f"mlaad/{folder}/{Path(r['path']).name}", split=split, label=1, family=folder,
                            source="mlaad", speaker=f"{base}:{voice}", parent=r["original_file"],
                            locator=f"{tp.name}::{folder}/{Path(r['path']).name}"))
    return out


def genuine_split(corpus, spk, part):
    if corpus in ("librittsr", "libritts"):
        return {"train.clean.100": "train", "dev.clean": "cal", "dev.other": "cal",
                "test.clean": "test", "test.other": "test"}[part]
    if corpus == "vctk":
        b = h("vctk/" + spk) % 10
        return "train" if b < 6 else ("cal" if b < 8 else "test")
    if corpus == "globe":
        return "test" if part == "test" else "train"
    if corpus == "mailabs":
        return "train"
    if corpus == "edacc":  # spk here is the conversation id (EDACC-C08-A/B share a recording)
        b = h("edacc/" + spk) % 5
        return "test" if b == 0 else ("cal" if b == 1 else "train")
    if corpus == "ami":  # spk here is the meeting/speaker connected component (see ami_components)
        b = h("ami/" + spk) % 5
        return "test" if b == 0 else ("cal" if b == 1 else "train")
    raise ValueError(corpus)


def audio_col(table):
    for name in table.column_names:
        v = table.column(name)[0].as_py()
        if isinstance(v, dict) and "bytes" in v:
            return name
    raise ValueError("no audio column")


def spk_col(cols):
    for c in ("speaker_id", "speaker", "client_id", "spk_id"):
        if c in cols:
            return c
    raise ValueError(f"no speaker column in {cols}")


def ami_components(raw):
    """Union meetings and speakers: a meeting (all mic views) and every meeting a speaker attends form one
    component, so no AMI recording or speaker spans two splits."""
    import pyarrow.parquet as pq

    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for p in sorted((raw / "ami").rglob("*.parquet")):
        t = pq.read_table(p, columns=["meeting_id", "speaker_id"])
        for m, sp in zip(t.column("meeting_id").to_pylist(), t.column("speaker_id").to_pylist()):
            a, b = find("m:" + str(m)), find("s:" + str(sp))
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {k: find(k) for k in list(parent)}


def iter_genuine(raw):
    """Yield (uid, corpus, split, speaker, parent, group, raw bytes) for all genuine sources.
    parent = recording identity (mic/channel siblings share it); group = the unit that decides the split."""
    import pyarrow.parquet as pq

    comp = ami_components(raw) if (raw / "ami").exists() else {}

    # LibriTTS-R and GLOBE v2 are excluded (neural restoration / enhancement + upsampling: not clean genuine).
    for corpus in ("libritts", "vctk", "ami", "edacc"):
        for p in sorted((raw / corpus).rglob("*.parquet")):
            part = p.parent.name if corpus == "libritts" else ("test" if p.name.startswith("test") else "train")
            if corpus == "ami":
                part = p.parent.name  # ihm (headset) / sdm (distant mic): recorded per row below
            pf = pq.ParquetFile(p)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg)
                cols = t.column_names
                ac, sc = audio_col(t), spk_col(cols)

                def col(c, t=t, cols=cols):
                    return t.column(c).to_pylist() if c in cols else None

                ids, text_id, meet, begin = col("id"), col("text_id"), col("meeting_id"), col("begin_time")
                for i, (a, s) in enumerate(zip(t.column(ac).to_pylist(), t.column(sc).to_pylist())):
                    s = str(s)
                    uid = f"{corpus}/{s}/{ids[i] if ids else (a.get('path') or f'{p.stem}-{rg}-{i}')}"
                    parent, group = uid, s
                    if corpus == "ami":
                        uid = f"ami-{part}/{s}/{ids[i] if ids else i}"
                        parent = f"ami/{meet[i]}/{s}/{begin[i]}"  # IHM and SDM views of one segment
                        group = comp["m:" + str(meet[i])]
                    elif corpus == "vctk" and text_id:
                        parent = f"vctk/{s}/{text_id[i]}"  # mic1/mic2 siblings of one utterance
                    elif corpus == "edacc":
                        group = s.rsplit("-", 1)[0]  # conversation
                    yield uid, corpus, genuine_split(corpus, group, part), s, parent, group, a["bytes"]


def iter_mailabs(tgz):
    with tarfile.open(tgz) as t:
        for m in t:
            if m.isfile() and m.name.endswith(".wav") and "/by_book/" in m.name:
                spk = m.name.split("/by_book/")[1].split("/")[1]
                yield (f"mailabs/{m.name}", "mailabs", "train", spk, f"mailabs/{m.name}", spk,
                       t.extractfile(m).read())


def hindi_split(spk):
    b = h(spk) % 10
    return "test" if b in (0, 1) else ("cal" if b == 2 else ("dev" if b == 3 else "train"))


def iter_hindi(root, kathbath_shards=8):
    """Kathbath + IndicVoices genuine and IndicSynth fakes (Drive parquet). IndicSynth target speakers are
    Kathbath speakers, so both share the 'hi-kb/' namespace and split by speaker together."""
    import pyarrow.parquet as pq

    srcs = [("kathbath", sorted((root / "kathbath/hindi").glob("*.parquet"))[:kathbath_shards]),
            ("indicvoices", sorted((root / "indicvoices/hindi").glob("*.parquet"))),
            ("indicsynth", sorted((root / "indicsynth/Hindi").glob("*.parquet")))]
    for corpus, files in srcs:
        for p in files:
            pf = pq.ParquetFile(p)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg)
                ac = audio_col(t)
                cols = t.column_names
                if corpus == "indicsynth":
                    spk = [f"hi-kb/{s}" for s in t.column("Target Speaker ID").to_pylist()]
                    gen = t.column("Generative Model").to_pylist()
                else:
                    pre = "hi-kb/" if corpus == "kathbath" else "hi-iv/"
                    spk = [pre + str(s) for s in t.column("speaker_id").to_pylist()]
                    gen = [None] * len(spk)
                for i, (a, s, g) in enumerate(zip(t.column(ac).to_pylist(), spk, gen)):
                    uid = f"{corpus}/{p.stem}/{rg}/{i}"
                    yield uid, corpus, hindi_split(s), s, g, a["bytes"]


def iter_voxenes(zp):
    """VoxENES 2026 'original' fakes only (VC: seed_vc, openvoice_v2, rvc_v2; TTS: 2025 systems), training/dev.
    Bona fide and the publisher's augmented copies are skipped (no speaker metadata; own augmentation used)."""
    import zipfile

    with zipfile.ZipFile(zp) as z:
        for n in sorted(z.namelist()):
            parts = n.split("/")
            if len(parts) < 5 or parts[2] != "original" or not n.endswith((".flac", ".wav", ".mp3")):
                continue
            kind, system = parts[1], parts[3]
            split = "dev" if h(n) % 10 == 0 else "train"
            yield dict(uid=f"voxenes/{n}", split=split, label=1, family=f"VoxENES/{kind}/{system}",
                       source="voxenes", speaker=f"voxenes:{system}", parent=f"voxenes/{n}", locator=""), z.read(n)


def iter_indiefake(tp):
    """Capped, speaker-split IndieFake subset (pack_indiefake manifest): genuine train/test, fakes train/test."""
    with tarfile.open(tp) as t:
        man = {m["member"]: m for m in json.loads(t.extractfile("indiefake/manifest.json").read())}
        for m in t:
            r = man.get(m.name)
            if r:
                yield r, t.extractfile(m).read()


def native_info(raw):
    """Original duration (s) and native sample rate before any treatment; (None, None) if unreadable."""
    try:
        import soundfile as sf

        i = sf.info(io.BytesIO(raw))
        return round(i.frames / i.samplerate, 3), int(i.samplerate)
    except Exception:
        return None, None


def _crop_job(args):
    uid, raw = args
    dur, sr = native_info(raw)
    try:
        return uid, crop(raw), None, dur, sr
    except Exception as e:
        return uid, None, f"error:{type(e).__name__}", dur, sr


class ShardWriter:
    def __init__(self, d, name, size=4000):
        self.d, self.name, self.size, self.buf, self.meta, self.n = d, name, size, [], [], 0
        d.mkdir(parents=True, exist_ok=True)

    def add(self, row, x):
        self.buf.append(x)
        self.meta.append(row)
        if len(self.buf) >= self.size:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        p = self.d / f"{self.name}-{self.n:04d}.npy"
        np.save(p, np.stack(self.buf))
        with (self.d / f"{self.name}-{self.n:04d}.jsonl").open("w") as f:
            for i, r in enumerate(self.meta):
                f.write(json.dumps(dict(r, shard=p.name, idx=i)) + "\n")
        self.buf, self.meta, self.n = [], [], self.n + 1


def prep(raw, work, mailabs_tgz, workers, hindi_root=None, include_voxenes=False):
    from multiprocessing import Pool

    stage = work / f"stage-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    shards = stage / "shards"
    excl = Counter()
    writers = {s: ShardWriter(shards, s) for s in ("train", "dev", "cal", "test")}
    rows = mlaad_rows(raw)
    print(f"mlaad rows {len(rows)} {Counter((r['split'], r['family'] in TARGETS) for r in rows)}", flush=True)
    seen_orig, kept = set(), []
    for r in sorted(rows, key=lambda r: key_uid(r["uid"])):
        k = (r["family"], r["parent"])
        if k in seen_orig:
            excl[f"{r['split']}|{r['family']}|repeated_original"] += 1
            continue
        seen_orig.add(k)
        kept.append(dict(r, group=r["parent"]))
    rows = kept
    by_tar = defaultdict(list)
    for r in rows:
        by_tar[r["locator"].split("::")[0]].append(r)
    with Pool(workers) as pool:
        for tname, rs in sorted(by_tar.items()):
            want = {r["locator"].split("::")[1]: r for r in rs}
            jobs = []
            with tarfile.open(raw / "mlaad" / tname) as t:
                for m in t:
                    if m.name in want:
                        jobs.append((want[m.name]["uid"], t.extractfile(m).read()))
            byuid = {r["uid"]: r for r in rs}
            for uid, x, err, dur, sr in pool.imap_unordered(_crop_job, jobs, chunksize=8):
                r = dict(byuid[uid], dur_s=dur, sr_native=sr)
                if x is None:
                    excl[f"{r['split']}|{r['family']}|{err or 'short'}"] += 1
                    continue
                writers[r["split"]].add(r, x)
            print(f"  {tname}: {len(jobs)}", flush=True)
        # genuine with caps (train) and per-speaker caps (cal/test), deterministic order by uid hash
        per_spk, per_corpus = Counter(), Counter()
        srcs = [iter_genuine(raw)] + ([iter_mailabs(mailabs_tgz)] if mailabs_tgz else [])

        def admit():
            for it in srcs:
                for uid, corpus, split, spk, parent, group, b in it:
                    cap = GEN_CAP_SPK if split == "train" else EVAL_CAP_SPK
                    if split == "train" and h(f"{corpus}/{group}") % 10 == 0:
                        split = "dev"
                    k = (corpus, split, spk)
                    if (corpus, parent) in seen_parent:  # one recording per condition (mic siblings dropped)
                        excl[f"{split}|genuine/{corpus}|duplicate_parent"] += 1
                        continue
                    if per_spk[k] >= cap or (split in ("train", "dev") and per_corpus[(corpus, split)] >= GEN_CAP_CORPUS):
                        continue
                    seen_parent.add((corpus, parent))
                    per_spk[k] += 1
                    per_corpus[(corpus, split)] += 1
                    meta[uid] = dict(uid=uid, split=split, label=0, family="genuine", source=corpus,
                                     speaker=f"{corpus}/{spk}", parent=f"{corpus}:{parent}",
                                     group=f"{corpus}:{group}", locator="")
                    yield uid, b

        meta, seen_parent = {}, set()
        itp = raw / "indiefake" / "indiefake.tar"

        def admit_all():
            yield from admit()
            if hindi_root:
                fake_n = Counter()
                for uid, corpus, split, spk, g, b in iter_hindi(hindi_root):
                    cap = GEN_CAP_SPK if split in ("train", "dev") else EVAL_CAP_SPK
                    k = (corpus, split, spk)
                    if per_spk[k] >= cap:
                        continue
                    if g is None and split in ("train", "dev") and per_corpus[(corpus, split)] >= GEN_CAP_CORPUS:
                        continue
                    if g is not None and (split == "cal" or (split in ("train", "dev") and fake_n[split] >= HINDI_CAP_FAKE)):
                        continue
                    per_spk[k] += 1
                    per_corpus[(corpus, split)] += 1
                    fake_n[split] += g is not None
                    meta[uid] = dict(uid=uid, split=split, label=int(g is not None),
                                     family=f"IndicSynth/{g}" if g else "genuine", source=corpus, speaker=spk,
                                     parent=uid, locator="")
                    yield uid, b
            vzp = raw / "voxenes" / "voxenes-2026.zip"
            if include_voxenes and vzp.exists():  # off by default: no source/target speaker lineage
                for r, b in iter_voxenes(vzp):
                    meta[r["uid"]] = r
                    yield r["uid"], b
            if itp.exists():
                for r, b in iter_indiefake(itp):
                    spk = "indiefake/" + r["speaker"]
                    split = r["split"]
                    if split == "train" and h(spk) % 10 == 0:
                        split = "dev"
                    elif split == "train" and h(spk) % 10 == 1:
                        split = "cal"  # calibration genuine only, capped like other eval speakers
                        if r["label"] == 1 or per_spk[("indiefake", "cal", spk)] >= EVAL_CAP_SPK:
                            continue
                        per_spk[("indiefake", "cal", spk)] += 1
                    uid = r["member"]
                    meta[uid] = dict(uid=uid, split=split, label=r["label"],
                                     family="IndieFake" if r["label"] else "genuine", source="indiefake",
                                     speaker=spk, parent=uid, locator="")
                    yield uid, b

        for uid, x, err, dur, sr in pool.imap_unordered(_crop_job, admit_all(), chunksize=8):
            r = dict(meta.pop(uid), dur_s=dur, sr_native=sr)
            if x is None:
                excl[f"{r['split']}|genuine/{r['source']}|{err or 'short'}"] += 1
                continue
            writers[r["split"]].add(r, x)
    for w in writers.values():
        w.flush()
    man = load_rows(shards)
    counts = Counter((r["split"], r["family"] if r["label"] else "genuine/" + r["source"]) for r in man)
    spk = defaultdict(set)
    for r in man:
        spk[(r["split"], r["family"] if r["label"] else "genuine/" + r["source"])].add(r["speaker"])
    rep = {"made": now(), "counts": {f"{a}|{b}": n for (a, b), n in sorted(counts.items())},
           "speakers_or_voices": {f"{a}|{b}": len(v) for (a, b), v in sorted(spk.items())},
           "excluded": dict(sorted(excl.items())), "checks": leakage(man),
           "coverage": coverage(man), "voxenes_included": include_voxenes,
           "hindi_root": str(hindi_root) if hindi_root else None}
    (stage / "prep.json").write_text(json.dumps(rep, indent=1))
    rep["prepared"] = publish_prepared(work, stage)
    return rep


def key_uid(u):
    return hashlib.sha256(u.encode()).hexdigest()


def load_rows(shards, splits=None):
    rows = []
    for p in sorted(shards.glob("*.jsonl")):
        if splits and p.name.split("-")[0] not in splits:
            continue
        rows += [json.loads(line) for line in p.read_text().splitlines()]
    return rows


def verify_shards(shards):
    """Row counts, row->audio references and hashes of every shard file. Returns (files, fingerprint)."""
    files, refs = {}, set()
    for j in sorted(shards.glob("*.jsonl")):
        npy = j.with_suffix(".npy")
        if not npy.exists():
            raise ValueError(f"{j.name}: missing audio shard {npy.name}")
        arr = np.load(npy, mmap_mode="r")
        rows = [json.loads(line) for line in j.read_text().splitlines()]
        if arr.ndim != 2 or arr.shape[1] != WIN or len(rows) != arr.shape[0]:
            raise ValueError(f"{j.name}: {len(rows)} rows vs audio shape {arr.shape}")
        for i, r in enumerate(rows):
            if r["shard"] != npy.name or r["idx"] != i:
                raise ValueError(f"{j.name}:{i} points at {r['shard']}:{r['idx']}")
            if (npy.name, i) in refs:
                raise ValueError(f"duplicate audio reference {npy.name}:{i}")
            refs.add((npy.name, i))
        files[j.name] = {"sha256": fsha(j), "rows": len(rows)}
        files[npy.name] = {"sha256": fsha(npy), "rows": int(arr.shape[0])}
    if not files:
        raise ValueError(f"no shards in {shards}")
    fp = hashlib.sha256("\n".join(f"{k} {v['sha256']}" for k, v in sorted(files.items())).encode()).hexdigest()
    return files, fp


def publish_prepared(work, stage):
    """Verify a staged prep, write its completion manifest last, then swap it in atomically.
    An existing prepared set is moved aside, never deleted."""
    files, fp = verify_shards(stage / "shards")
    leakage(load_rows(stage / "shards"))
    man = {"complete": True, "made": now(), "fingerprint": fp, "files": files, "code_sha256": fsha(__file__),
           "prep_json_sha256": fsha(stage / "prep.json")}
    (stage / "PREPARED.json").write_text(json.dumps(man, indent=1))
    dst = work / PREPARED
    if dst.exists():
        os.replace(dst, work / f"{PREPARED}-superseded-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}")
    os.replace(stage, dst)
    return {"fingerprint": fp, "files": len(files)}


def verify_prepared(work):
    """Fail clearly unless work/prepared is a complete, untouched prep (manifest, hashes, references)."""
    d = work / PREPARED
    mp = d / "PREPARED.json"
    if not mp.exists():
        raise ValueError(f"{d}: no completion manifest (interrupted copy or unpublished prep)")
    man = json.loads(mp.read_text())
    files, fp = verify_shards(d / "shards")
    if not man.get("complete") or fp != man["fingerprint"] or files != man["files"]:
        raise ValueError(f"{d}: shards differ from their completion manifest")
    return fp


def coverage(man):
    """Eligible recordings, unique parents and genuine speakers per split and family (preflight input)."""
    cov = defaultdict(lambda: {"recordings": 0, "parents": set(), "speakers": set()})
    for r in man:
        k = f"{r['split']}|{r['family'] if r['label'] else 'genuine'}"
        if r["label"] == 0 and r["source"] in HINDI_SOURCES:
            k = f"{r['split']}|genuine_hindi"
        c = cov[k]
        c["recordings"] += 1
        c["parents"].add(r["parent"])
        c["speakers"].add(r["speaker"])
    return {k: {"recordings": v["recordings"], "parents": len(v["parents"]), "speakers": len(v["speakers"])}
            for k, v in sorted(cov.items())}


def preflight(work, splits=("train", "dev", "cal", "test")):
    """Refuse to train/select/evaluate unless every target family and the English genuine controls meet the
    documented minimum coverage in each split."""
    cov = coverage(load_manifest(work))
    fails = []
    for sp in splits:
        for f in TARGETS:
            if sp in MIN_FAKE:
                n = cov.get(f"{sp}|{f}", {}).get("parents", 0)
                if n < MIN_FAKE[sp]:
                    fails.append(f"{sp}|{f}: {n} unique parents < {MIN_FAKE[sp]}")
        if sp in MIN_GEN:
            g = cov.get(f"{sp}|genuine", {"recordings": 0, "speakers": 0})
            if g["recordings"] < MIN_GEN[sp][0] or g["speakers"] < MIN_GEN[sp][1]:
                fails.append(f"{sp}|genuine: {g['recordings']} rec / {g['speakers']} spk < {MIN_GEN[sp]}")
    rep = {"checked": now(), "coverage": cov, "minimums": {"fake": MIN_FAKE, "genuine": MIN_GEN}, "fails": fails}
    (work / "preflight.json").write_text(json.dumps(rep, indent=1))
    if fails:
        raise ValueError("Coverage preflight failed: " + "; ".join(fails))
    return rep


def load_manifest(work, splits=None):
    return load_rows(work / PREPARED / "shards", splits)


def leakage(man):
    def sets(split, key, label=None):
        return {r[key] for r in man if r["split"] == split and (label is None or r["label"] == label)}

    dup = [u for u, n in Counter(r["uid"] for r in man).items() if n > 1]
    if dup:
        raise ValueError(f"duplicate recording ids: {dup[:5]}")
    rep_orig = [k for k, n in Counter((r["family"], r["parent"]) for r in man if r["label"] == 1).items() if n > 1]
    if rep_orig:
        raise ValueError(f"repeated originals within a family: {rep_orig[:5]}")
    split_of = defaultdict(set)
    for r in man:
        split_of[("parent", r["parent"])].add(r["split"])
        split_of[("group", r.get("group") or r["parent"])].add(r["split"])
        if r["label"] == 0:
            split_of[("genuine_speaker", r["speaker"])].add(r["split"])
    crossing = Counter(k[0] for k, v in split_of.items() if len(v) > 1)
    tr = {"train", "dev"}
    tr_par = set().union(*(sets(s, "parent", 1) for s in tr))
    tr_spk = set().union(*(sets(s, "speaker", 0) for s in tr))
    chk = {
        "test_fake_parent_in_train": len(sets("test", "parent", 1) & tr_par),
        "test_genuine_speaker_in_train": len(sets("test", "speaker", 0) & tr_spk),
        "cal_genuine_speaker_in_train": len(sets("cal", "speaker", 0) & tr_spk),
        "cal_test_genuine_speaker_overlap": len(sets("cal", "speaker", 0) & sets("test", "speaker", 0)),
    }
    # stock voices: compare voice names across folders of the same vendor
    def voices(split):
        return {(base_family(r["family"]).split("-")[0], r["speaker"].split(":", 1)[1]) for r in man
                if r["split"] == split and r["label"] == 1 and base_family(r["family"]).startswith(VOICE_SPLIT)}
    chk["test_stock_voice_in_train"] = len(voices("test") & (voices("train") | voices("dev")))
    chk.update({f"split_crossing_{k}": crossing.get(k, 0) for k in ("parent", "group", "genuine_speaker")})
    if any(chk.values()):
        raise ValueError(f"Split leakage: {chk}")
    return chk


# ------------------------------------------------------------------------- data access
class Store:
    def __init__(self, work, splits):
        self.rows = load_manifest(work, splits)
        self.work = work
        self._m = {}

    def x(self, r):
        p = self.work / PREPARED / "shards" / r["shard"]
        if p not in self._m:
            self._m[p] = np.load(p, mmap_mode="r")
        return self._m[p][r["idx"]].astype(np.float32) / 32768


def ffmpeg_codec(x, name):
    enc = {"mulaw8k": (["-ar", "8000", "-c:a", "pcm_mulaw", "-f", "wav"]),
           "alaw8k": (["-ar", "8000", "-c:a", "pcm_alaw", "-f", "wav"]),
           "gsm": (["-ar", "8000", "-c:a", "libgsm", "-f", "gsm"]),
           "g726": (["-ar", "8000", "-c:a", "g726", "-b:a", "32k", "-f", "wav"]),
           "opus8": (["-c:a", "libopus", "-b:a", "8k", "-f", "ogg"]),
           "opus12": (["-c:a", "libopus", "-b:a", "12k", "-f", "ogg"]),
           "opus16": (["-c:a", "libopus", "-b:a", "16k", "-f", "ogg"]),
           "opus24": (["-c:a", "libopus", "-b:a", "24k", "-f", "ogg"]),
           "mp3": (["-c:a", "libmp3lame", "-b:a", "32k", "-f", "mp3"])}[name]
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    a = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                        "-ac", "1"] + enc + ["pipe:1"], input=pcm, capture_output=True, check=True).stdout
    b = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ar", str(SR), "-ac", "1", "-f", "s16le",
                        "pipe:1"], input=a, capture_output=True, check=True).stdout
    y = np.frombuffer(b, "<i2").astype(np.float32) / 32768
    return np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]


def channel(x, name):
    return x if name == "clean" else ffmpeg_codec(x, name)


def augment(x, rng):
    from scipy.signal import butter, sosfilt

    if rng.random() < 0.3:
        sos = butter(8, rng.uniform(4000, 7800), "low", fs=SR, output="sos")
        x = sosfilt(sos, x).astype(np.float32)
    if rng.random() < 0.4:
        try:
            x = ffmpeg_codec(x, TRAIN_CODECS[rng.integers(len(TRAIN_CODECS))])
        except subprocess.CalledProcessError:
            pass
    x = x * (10 ** (rng.uniform(-6, 6) / 20))
    if rng.random() < 0.2:
        x = x + rng.normal(0, max(float(np.std(x)), 1e-4) * rng.uniform(0.01, 0.05), len(x))
    return np.clip(x, -1, 1).astype(np.float32)


def _stream_base():
    from torch.utils.data import IterableDataset

    return IterableDataset


class TrainStream(_stream_base()):
    """Picklable infinite stream: balanced group sampling + label-symmetric augmentation."""

    def __init__(self, work, groups, probs):
        self.work, self.groups, self.probs = work, groups, probs

    def __iter__(self):
        import torch

        store = Store(self.work, {"train"})
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(1000 + (wi.id if wi else 0) + int(time.time() * 1000) % 100000)
        while True:
            g = self.groups[rng.choice(len(self.groups), p=self.probs)]
            r = store.rows[g[rng.integers(len(g))]]
            yield torch.from_numpy(augment(store.x(r), rng)), float(r["label"])


# ------------------------------------------------------------------------- model
def build_model(encoder, freeze_below=0):
    import torch
    from torch import nn
    from transformers import AutoModel

    class Det(nn.Module):
        def __init__(self):
            super().__init__()
            self.ssl = AutoModel.from_pretrained(encoder, layerdrop=0.0)
            self.ssl.feature_extractor._freeze_parameters()
            if freeze_below:
                for mod in [self.ssl.feature_projection] + list(self.ssl.encoder.layers[:freeze_below]):
                    mod.requires_grad_(False)
            n = self.ssl.config.num_hidden_layers + 1
            self.w = nn.Parameter(torch.zeros(n))
            d = self.ssl.config.hidden_size
            self.head = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))

        def forward(self, x):
            x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-5)
            hs = torch.stack(self.ssl(x, output_hidden_states=True).hidden_states, 0)
            hcat = (torch.softmax(self.w, 0)[:, None, None, None] * hs).sum(0)
            return self.head(torch.cat([hcat.mean(1), hcat.std(1)], -1)).squeeze(-1)

    return Det()


def train(work, drive, encoder, steps, batch, workers, lr_enc, lr_head, eval_every, tag="c2", freeze_below=0,
          stop_after=None, skip_preflight=False):
    import torch
    from torch.utils.data import DataLoader, IterableDataset

    fp = verify_prepared(work)
    if not skip_preflight:
        preflight(work, ("train", "dev"))
    cfg = {"dataset_fingerprint": fp, "tag": tag, "encoder": encoder, "freeze_below": freeze_below,
           "steps": steps, "batch": batch, "lr_enc": lr_enc, "lr_head": lr_head, "eval_every": eval_every,
           "code_sha256": fsha(__file__)}
    store = Store(work, {"train"})
    dev = Store(work, {"dev"})
    rows = store.rows
    # sampling weights: 50% genuine (equal per corpus), 50% fake (half to target + related, half diverse)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if r["label"] == 0:
            groups[("g", r["source"])].append(i)
        else:
            b = base_family(r["family"])
            tgt = b in TARGETS or any(b in v for v in RELATED.values())
            hindi = r["family"].startswith(("hi__", "IndicSynth/"))
            vc = r["family"].startswith("VoxENES/vc/") or b == "RVC"
            groups[("h" if hindi else ("v" if vc else ("t" if tgt else "d")), r["family"])].append(i)
    kinds = {"g": 0.5, "t": 0.26, "d": 0.12, "h": 0.07, "v": 0.05}
    kinds = {k: v for k, v in kinds.items() if any(kk[0] == k for kk in groups)}
    keys = list(groups)
    probs = np.array([kinds[k[0]] / sum(1 for kk in keys if kk[0] == k[0]) for k in keys])
    probs /= probs.sum()

    ds = TrainStream(work, [groups[k] for k in keys], probs)
    dl = iter(DataLoader(ds, batch_size=batch, num_workers=workers, prefetch_factor=4, persistent_workers=True))
    dev_rows = dev.rows
    model = build_model(encoder, freeze_below).cuda()
    enc_params = [p for n, p in model.named_parameters() if n.startswith("ssl.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": lr_enc},
                             {"params": list(model.head.parameters()) + [model.w], "lr": lr_head}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_enc, lr_head], total_steps=steps, pct_start=0.08)
    ck = work / f"ckpt-{tag}"
    ck.mkdir(exist_ok=True)
    start, best, hist = 0, None, []
    if (ck / "last.pt").exists():
        s = torch.load(ck / "last.pt", map_location="cpu", weights_only=False)
        diff = {k: (s.get("cfg", {}).get(k), v) for k, v in cfg.items() if k != "code_sha256" and s.get("cfg", {}).get(k) != v}
        if diff:
            raise ValueError(f"{ck}/last.pt belongs to a different dataset or candidate config: {diff}")
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        start, best, hist = s["step"], s["best"], s["hist"]
        print(f"resumed at {start}", flush=True)
    lossf = torch.nn.BCEWithLogitsLoss()
    t0 = time.time()
    model.train()
    for step in range(start, steps):
        x, y = next(dl)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x.cuda(non_blocking=True))
        loss = lossf(out.float(), y.cuda().float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            sc = score_rows(model, dev, dev_rows, "clean")
            m = dev_metric(dev_rows, sc)
            hist.append({"step": step + 1, **m})
            print(f"DEV step {step + 1}: {json.dumps(m)}", flush=True)
            if m["sufficient"] and (best is None or m["key"] > best["key"]):
                best = {"step": step + 1, **m}
                torch.save(model.state_dict(), ck / "best.tmp")
                os.replace(ck / "best.tmp", ck / "best.pt")
                (ck / "best.json").write_text(json.dumps({"cfg": cfg, "step": step + 1, "metric": m,
                                                          "sha256": fsha(ck / "best.pt")}, indent=1))
            model.train()
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step + 1, "best": best, "hist": hist, "cfg": cfg}, ck / "last.tmp")
            os.replace(ck / "last.tmp", ck / "last.pt")
            (work / f"train-{tag}.json").write_text(json.dumps({"tag": tag, "encoder": encoder, "best": best,
                                                                "hist": hist}, indent=1))
            mirror(work, drive, [f"train-{tag}.json", f"ckpt-{tag}/best.pt", f"ckpt-{tag}/best.json",
                                 f"ckpt-{tag}/last.pt"])
            if stop_after and step + 1 >= stop_after and step + 1 < steps:
                print(f"STOP_AFTER {step + 1}: simulated interruption", flush=True)
                return {"stopped_at": step + 1, "best": best}
    return best


def score_rows(model, store, rows, ch, bs=int(os.environ.get("V290_EVAL_BS", "64"))):
    import torch

    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(rows), bs):
            xs = [channel(store.x(r), ch) for r in rows[i:i + bs]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out += model(torch.from_numpy(np.stack(xs)).cuda()).float().cpu().tolist()
    return out


def cutoff_at(gen_scores, fa=0.15):
    s = sorted(gen_scores, reverse=True)
    k = int(len(s) * fa)
    return math.nextafter(s[k], math.inf) if k < len(s) else -math.inf


def dev_metric(rows, sc, min_fake=None):
    """Development selection metric. Insufficient (key = -1) unless every target family has at least
    min_fake recordings and English genuine controls exist: absent families never count as perfect."""
    min_fake = MIN_FAKE["dev"] if min_fake is None else min_fake
    gen = [s for r, s in zip(rows, sc) if r["label"] == 0 and r.get("source") not in HINDI_SOURCES]
    c = cutoff_at(gen) if gen else math.inf
    det, n, short = {}, {}, []
    for f in TARGETS:
        v = [s >= c for r, s in zip(rows, sc) if r["family"] == f]
        n[f] = len(v)
        det[f] = round(sum(v) / len(v), 4) if v else None
        if len(v) < min_fake:
            short.append(f"{f}: {len(v)} < {min_fake}")
    if not gen:
        short.append("no English genuine")
    allf = [s >= c for r, s in zip(rows, sc) if r["label"] == 1]
    ok = not short
    return {"cutoff": c, "det": det, "n": n, "pooled": round(sum(allf) / len(allf), 4) if allf else 0.0,
            "sufficient": ok, "insufficient": short, "key": min(det.values()) if ok else -1.0}


def mirror(work, drive, rels):
    if not drive:
        return
    for rel in rels:
        src, dst = work / rel, drive / rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            tmp.write_bytes(src.read_bytes())
            os.replace(tmp, dst)


# ------------------------------------------------------------------------- shortcut audit
def simple_features(x):
    from scipy.signal import welch

    f, p = welch(x, SR, nperseg=1024)
    p = p + 1e-12
    cum = np.cumsum(p) / p.sum()
    frames = x[: len(x) // 400 * 400].reshape(-1, 400)
    e = 10 * np.log10((frames ** 2).mean(1) + 1e-10)
    return {"rms_db": float(10 * np.log10((x ** 2).mean() + 1e-10)),
            "rolloff95": float(f[np.searchsorted(cum, 0.95)]),
            "upper_band": float(p[f > 4000].sum() / p.sum()),
            "tilt": float(np.polyfit(np.log(f[1:]), 10 * np.log10(p[1:]), 1)[0]),
            "speech_ratio": float((e > e.max() - 30).mean()),
            "zcr": float((np.diff(np.sign(x)) != 0).mean())}


def audit(work):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    res = {}
    for split in ("train", "dev"):
        st = Store(work, {split})
        rng = random.Random(7)
        rows = st.rows if len(st.rows) < 12000 else rng.sample(st.rows, 12000)
        res[split] = (rows, [dict(simple_features(st.x(r)),
                                  native_duration=float(r.get("dur_s") or -1),
                                  native_sample_rate=float(r.get("sr_native") or -1)) for r in rows])
    names = list(res["train"][1][0])
    out = {}
    for fam in TARGETS + ["pooled"]:
        def sel(split):
            rows, feats = res[split]
            idx = [i for i, r in enumerate(rows) if r["label"] == 0 or fam == "pooled" or r["family"] == fam]
            return np.array([[feats[i][n] for n in names] for i in idx]), np.array([rows[i]["label"] for i in idx])

        Xa, ya = sel("train")
        Xb, yb = sel("dev")
        if len(set(yb)) < 2:
            continue
        o = {}
        for j, n in enumerate(names + ["joint"]):
            cols = list(range(len(names))) if n == "joint" else [j]
            sc = StandardScaler().fit(Xa[:, cols])
            m = LogisticRegression(max_iter=1000).fit(sc.transform(Xa[:, cols]), ya)
            auc = roc_auc_score(yb, m.predict_proba(sc.transform(Xb[:, cols]))[:, 1])
            o[n] = round(max(auc, 1 - auc), 4)
        out[fam] = o
    flags = {f: [n for n, a in o.items() if a > 0.80] for f, o in out.items()}
    rep = {"made": now(), "oriented_auc_train_to_dev": out, "red_flags_gt_0.80": flags,
           "rule": "protocol 28 s5: non-blocking; >0.80 red flag, 0.60-0.80 reported"}
    (work / "audit.json").write_text(json.dumps(rep, indent=1))
    return rep


# ------------------------------------------------------------------------- calibrate / signoff / evaluate
def load_member(work, tag, encoder):
    import torch

    m = build_model(encoder).cuda()
    meta = json.loads((work / f"ckpt-{tag}/best.json").read_text())
    if meta["cfg"]["dataset_fingerprint"] != verify_prepared(work):
        raise ValueError(f"ckpt-{tag} was trained on a different prepared dataset")
    if meta["sha256"] != fsha(work / f"ckpt-{tag}/best.pt") or meta["cfg"]["encoder"] != encoder:
        raise ValueError(f"ckpt-{tag}/best.pt does not match its binding record")
    m.load_state_dict(torch.load(work / f"ckpt-{tag}/best.pt", map_location="cpu"))
    return m.eval()


def zstats(v):
    return float(np.mean(v)), float(np.std(v) or 1.0)


def select(work, cands):
    """Prespecified (protocol 28 amendment 3): score development with each fine-tuned candidate and the
    equal-weight z-mean of all; z uses development genuine mean/sd. Pick max of min-family detection at the
    development 15% cutoff; ties -> higher pooled detection -> fewer members."""
    dev = Store(work, {"dev"})
    rows = dev.rows
    gen = [i for i, r in enumerate(rows) if r["label"] == 0]
    members, z = [], {}
    for tag, enc in cands:
        m = load_member(work, tag, enc)
        sc = np.array(score_rows(m, dev, rows, "clean"))
        mu, sd = zstats(sc[gen])
        z[tag] = (sc - mu) / sd
        members.append({"tag": tag, "encoder": enc, "sha256": fsha(work / f"ckpt-{tag}/best.pt"), "mu": mu, "sd": sd})
        del m
    options = [[t] for t, _ in cands] + ([[t for t, _ in cands]] if len(cands) > 1 else [])
    table = []
    for opt in options:
        sc = np.mean([z[t] for t in opt], 0).tolist()
        mtr = dev_metric(rows, sc)
        table.append({"members": opt, **mtr})
    best = max(table, key=lambda r: (r["key"], r["pooled"], -len(r["members"])))
    if not best["sufficient"]:
        raise ValueError(f"Development coverage insufficient for selection: {best['insufficient']}")
    sel = {"made": now(), "rule": select.__doc__, "options": table, "primary": best["members"],
           "members": [m for m in members if m["tag"] in best["members"]]}
    (work / "selection.json").write_text(json.dumps(sel, indent=1))
    return sel


class Scorer:
    def __init__(self, work):
        self.sel = json.loads((work / "selection.json").read_text())
        self.ms = []
        for m in self.sel["members"]:
            if fsha(work / f"ckpt-{m['tag']}/best.pt") != m["sha256"]:
                raise ValueError(f"member {m['tag']} changed after selection")
            self.ms.append((load_member(work, m["tag"], m["encoder"]), m["mu"], m["sd"]))
        self.sha = fsha(work / "selection.json")

    def __call__(self, store, rows, ch):
        return np.mean([(np.array(score_rows(m, store, rows, ch)) - mu) / sd for m, mu, sd in self.ms], 0).tolist()


def calibrate(work, encoder=None):
    st = Store(work, {"cal"})
    gen = [r for r in st.rows if r["label"] == 0 and r["source"] not in HINDI_SOURCES]  # English cutoff
    scorer = Scorer(work)
    sc = scorer(st, gen, "clean")
    c = cutoff_at(sc)
    by = Counter(r["source"] for r in gen)
    fl = Counter(r["source"] for r, s in zip(gen, sc) if s >= c)
    rep = {"made": now(), "model_sha256": scorer.sha, "primary": scorer.sel["primary"], "cutoff": c,
           "rule": "lowest cutoff with <= floor(0.15 n) calibration genuine flagged (clean); flag if score >= cutoff",
           "flagged": f"{sum(s >= c for s in sc)}/{len(sc)}", "by_source": {k: f"{fl[k]}/{n}" for k, n in by.items()},
           "speakers": len({r["speaker"] for r in gen})}
    (work / "cutoff.json").write_text(json.dumps(rep, indent=1))
    return rep


TEST_FIELDS = ("uid", "split", "label", "family", "source", "speaker", "parent", "group", "shard", "idx")


def test_binding(work):
    """Hash every field that decides which audio is scored, plus the bytes of each referenced shard."""
    rows = sorted(load_manifest(work, {"test"}), key=lambda r: r["uid"])
    if len({r["uid"] for r in rows}) != len(rows) or len({(r["shard"], r["idx"]) for r in rows}) != len(rows):
        raise ValueError("duplicate test recording or audio reference")
    sh = work / PREPARED / "shards"
    shards = {name: fsha(sh / name) for name in sorted({r["shard"] for r in rows})}
    blob = "\n".join(json.dumps({k: r.get(k) for k in TEST_FIELDS}, sort_keys=True) for r in rows)
    blob += "\n" + json.dumps(shards, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest(), rows, shards


def atomic_write(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def signoff(work, protocol, commit):
    """Write-once: binds prepared data, selected weights, preprocessing code, fixed cutoff and the test set."""
    p = work / "signoff.json"
    if p.exists():
        raise SystemExit("signoff exists (write-once)")
    if (work / "eval-scores.jsonl").exists() or (work / "result.json").exists():
        raise ValueError("test already scored")
    fp = verify_prepared(work)
    sel = json.loads((work / "selection.json").read_text())
    for m in sel["members"]:
        if fsha(work / f"ckpt-{m['tag']}/best.pt") != m["sha256"]:
            raise ValueError(f"weights of {m['tag']} changed after selection")
    cut = json.loads((work / "cutoff.json").read_text())
    if cut["model_sha256"] != fsha(work / "selection.json"):
        raise ValueError("model changed after calibration")
    tb, rows, shards = test_binding(work)
    rec = {"signed": now(), "protocol_sha256": fsha(protocol), "code_commit": commit, "code_sha256": fsha(__file__),
           "dataset_fingerprint": fp, "selection_sha256": fsha(work / "selection.json"),
           "member_weights_sha256": {m["tag"]: m["sha256"] for m in sel["members"]},
           "model_sha256": cut["model_sha256"], "cutoff": cut["cutoff"], "cutoff_sha256": fsha(work / "cutoff.json"),
           "test_binding_sha256": tb, "test_shards_sha256": shards,
           "test_rows": dict(Counter(r["family"] if r["label"] else "genuine/" + r["source"] for r in rows))}
    atomic_write(p, json.dumps(rec, indent=1))
    return rec


def check_signoff(work):
    s = json.loads((work / "signoff.json").read_text())
    if verify_prepared(work) != s["dataset_fingerprint"]:
        raise ValueError("prepared data changed after signoff")
    if fsha(work / "selection.json") != s["selection_sha256"] or fsha(work / "cutoff.json") != s["cutoff_sha256"]:
        raise ValueError("selection or cutoff changed after signoff")
    for tag, h_ in s["member_weights_sha256"].items():
        if fsha(work / f"ckpt-{tag}/best.pt") != h_:
            raise ValueError(f"weights of {tag} changed after signoff")
    tb, rows, _ = test_binding(work)
    if tb != s["test_binding_sha256"]:
        raise ValueError("test rows or their audio changed after signoff")
    return s, rows


def wilson(k, n, z=1.959964):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(m - hw, 4), round(m + hw, 4)]


def boot(cl, B=2000, seed=20261126):
    cl = list(cl.values())
    if len(cl) < 2:
        return None
    rng = random.Random(seed)
    v = []
    for _ in range(B):
        k = n = 0
        for _ in cl:
            c = cl[rng.randrange(len(cl))]
            k += sum(c); n += len(c)
        v.append(k / n)
    v.sort()
    return [round(v[int(0.025 * B)], 4), round(v[int(0.975 * B) - 1], 4)]


def evaluate(work, encoder=None, scorer=None, channels=None):
    """One pass on the signed-off test set. A completed evaluation bound to the same signoff is preserved and
    returned unchanged on retry; outputs are published atomically."""
    sig = fsha(work / "signoff.json")
    if (work / "result.json").exists():
        prev = json.loads((work / "result.json").read_text())
        if prev.get("signoff_sha256") == sig:
            print("evaluation already complete for this signoff; preserved", flush=True)
            return prev
        raise ValueError("result.json exists for a different signoff; refusing to overwrite")
    s, rows = check_signoff(work)
    st = Store(work, {"test"})
    scorer = scorer or Scorer(work)
    c = s["cutoff"]
    res = {"signoff_sha256": sig, "evaluated": now(), "cutoff": c, "channels": {}}
    lines = []
    for ch in (channels or CHANNELS_EVAL):
        if True:
            sc = scorer(st, rows, ch)
            for r, v in zip(rows, sc):
                lines.append(json.dumps({"uid": r["uid"], "channel": ch, "score": v}))
            cells = defaultdict(lambda: defaultdict(list))
            for r, v in zip(rows, sc):
                hindi_gen = not r["label"] and r["source"] in HINDI_SOURCES
                cells[r["family"] if r["label"] else ("genuine_hindi" if hindi_gen else "genuine")][r["speaker"]].append(int(v >= c))
                if not r["label"]:
                    cells["genuine/" + r["source"]][r["speaker"]].append(int(v >= c))
                    if r["source"] in ("libritts", "vctk"):
                        cells["genuine_read_only"][r["speaker"]].append(int(v >= c))
            t = {}
            for k, cl in sorted(cells.items()):
                flat = [x for v in cl.values() for x in v]
                t[k] = {"flagged": sum(flat), "n": len(flat), "rate": round(sum(flat) / len(flat), 4),
                        "wilson95": wilson(sum(flat), len(flat)), "clusters": len(cl), "cluster_boot95": boot(cl)}
            g = t.get("genuine", {"flagged": 0, "n": 0, "wilson95": [0, 1], "cluster_boot95": None, "clusters": 0})
            small = [f for f in TARGETS if f in t and t[f]["n"] < MIN_FAKE["test"]]
            if g["n"] < MIN_GEN["test"][0] or g["clusters"] < MIN_GEN["test"][1]:
                small.append("genuine")
            gaps = {}
            if g["flagged"] > math.floor(0.15 * g["n"]):
                gaps["genuine_FA"] = g["flagged"] - math.floor(0.15 * g["n"])
            untested = [f for f in TARGETS if f not in t]
            for fam in TARGETS:
                if fam in untested:
                    continue
                need = math.ceil(0.90 * t[fam]["n"] - 1e-9)
                if t[fam]["flagged"] < need:
                    gaps[fam] = need - t[fam]["flagged"]
            lo_ok = all((t[f]["cluster_boot95"] or t[f]["wilson95"])[0] >= 0.90 for f in TARGETS if f in t) and \
                (g["cluster_boot95"] or g["wilson95"])[1] <= 0.15
            if untested or small:
                verdict = "INSUFFICIENT: " + ", ".join([f"untested {f}" for f in untested] + [f"undersized {f}" for f in small])
            else:
                verdict = "MISS" if gaps else ("PASS" if lo_ok else "PASS at point estimate; not robust")
            res["channels"][ch] = {"cells": t, "gaps_in_recordings": gaps, "verdict": verdict}
            print(ch, verdict, {k: f"{v['flagged']}/{v['n']}" for k, v in t.items()}, flush=True)
    audit = work / "audit.json"
    res["shortcut_audit"] = json.loads(audit.read_text()).get("red_flags_gt_0.80") if audit.exists() else "not run"
    atomic_write(work / "eval-scores.jsonl", "\n".join(lines) + "\n")
    atomic_write(work / "result.json", json.dumps(res, indent=1))
    return res


def baseline(work):
    """Release v1 on the same eligible test recordings: shipped cutoff and the same independent calibration
    rule (lowest cutoff with <= floor(0.15 n) English calibration genuine flagged). Run after signoff."""
    from detector.inference import Detector

    s, rows = check_signoff(work)
    det = Detector()
    st_t, st_c = Store(work, {"test"}), Store(work, {"cal"})
    cal = [r for r in st_c.rows if r["label"] == 0 and r["source"] not in HINDI_SOURCES]
    p = lambda st, r: float(det.analyze_wave(st.x(r).astype(np.float32))["fake_probability"])
    pc = [p(st_c, r) for r in cal]
    cut_cal = cutoff_at(pc)
    pt = [p(st_t, r) for r in rows]
    out = {"made": now(), "signoff_sha256": fsha(work / "signoff.json"), "release_sha256": det.release_sha256,
           "view": "same treated 4 s crop as the candidate", "cutoffs": {}}
    for name, c in (("shipped", SHIPPED_RELEASE_CUTOFF), ("calibrated", cut_cal)):
        cells = defaultdict(lambda: [0, 0])
        for r, v in zip(rows, pt):
            k = r["family"] if r["label"] else ("genuine_hindi" if r["source"] in HINDI_SOURCES else "genuine")
            for kk in [k] + ([f"genuine/{r['source']}"] if not r["label"] else []):
                cells[kk][0] += int(v >= c)
                cells[kk][1] += 1
        out["cutoffs"][name] = {"cutoff": c, "cells": {k: {"flagged": a, "n": b, "rate": round(a / b, 4)}
                                                      for k, (a, b) in sorted(cells.items())}}
    atomic_write(work / "baseline.json", json.dumps(out, indent=1))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["prep", "verify", "preflight", "audit", "train", "select", "calibrate",
                                      "signoff", "evaluate", "baseline"])
    p.add_argument("--include-voxenes", action="store_true", help="off for the primary run (no speaker lineage)")
    p.add_argument("--stop-after", type=int, help="smoke test: stop after this step to test resume")
    p.add_argument("--skip-preflight", action="store_true", help="smoke test on a tiny dataset only")
    p.add_argument("--tag", default="c2")
    p.add_argument("--freeze-below", type=int, default=0)
    p.add_argument("--cands", nargs="+", help="tag=encoder ...")
    p.add_argument("--raw", type=Path)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--drive", type=Path)
    p.add_argument("--mailabs", type=Path)
    p.add_argument("--hindi", type=Path, help="Drive development/ root holding kathbath, indicvoices, indicsynth")
    p.add_argument("--encoder", default="facebook/wav2vec2-xls-r-300m")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--lr-enc", type=float, default=2e-5)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--protocol", type=Path)
    p.add_argument("--commit")
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    if a.phase == "prep":
        r = prep(a.raw, a.work, a.mailabs, a.workers, a.hindi, a.include_voxenes)
    elif a.phase == "verify":
        r = {"fingerprint": verify_prepared(a.work)}
    elif a.phase == "preflight":
        verify_prepared(a.work)
        r = preflight(a.work)
    elif a.phase == "audit":
        r = audit(a.work)
    elif a.phase == "train":
        r = train(a.work, a.drive, a.encoder, a.steps, a.batch, a.workers, a.lr_enc, a.lr_head, a.eval_every,
                  a.tag, a.freeze_below, a.stop_after, a.skip_preflight)
    elif a.phase == "select":
        r = select(a.work, [tuple(c.split("=", 1)) for c in a.cands])
    elif a.phase == "calibrate":
        r = calibrate(a.work, a.encoder)
    elif a.phase == "signoff":
        r = signoff(a.work, a.protocol, a.commit)
    elif a.phase == "baseline":
        r = baseline(a.work)
    else:
        r = evaluate(a.work, a.encoder)
    print(json.dumps(r, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
