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
            meta = t.extractfile(f"{folder}/meta.csv").read().decode()
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
    if corpus == "librittsr":
        return {"train.clean.100": "train", "dev.clean": "cal", "dev.other": "cal",
                "test.clean": "test", "test.other": "test"}[part]
    if corpus == "vctk":
        b = h("vctk/" + spk) % 10
        return "train" if b < 6 else ("cal" if b < 8 else "test")
    if corpus == "globe":
        return "test" if part == "test" else "train"
    if corpus == "mailabs":
        return "train"
    if corpus == "ami":  # by speaker across official folds, so no AMI speaker spans train and test
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


def iter_genuine(raw):
    """Yield (uid, corpus, split, speaker, raw bytes) for all genuine sources."""
    import pyarrow.parquet as pq

    for corpus in ("librittsr", "vctk", "globe", "ami"):
        for p in sorted((raw / corpus).rglob("*.parquet")):
            part = p.parent.name if corpus == "librittsr" else ("test" if p.name.startswith("test") else "train")
            if corpus == "ami":
                part = p.parent.name  # ihm (headset) / sdm (distant mic): recorded per row below
            pf = pq.ParquetFile(p)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg)
                ac, sc = audio_col(t), spk_col(t.column_names)
                ids = t.column("id").to_pylist() if "id" in t.column_names else None
                for i, (a, s) in enumerate(zip(t.column(ac).to_pylist(), t.column(sc).to_pylist())):
                    s = str(s)
                    uid = f"{corpus}/{s}/{ids[i] if ids else (a.get('path') or f'{p.stem}-{rg}-{i}')}"
                    if corpus == "ami":
                        uid = f"ami-{part}/{s}/{ids[i] if ids else i}"
                    yield uid, corpus, genuine_split(corpus, s, part), s, a["bytes"]


def iter_mailabs(tgz):
    with tarfile.open(tgz) as t:
        for m in t:
            if m.isfile() and m.name.endswith(".wav") and "/by_book/" in m.name:
                spk = m.name.split("/by_book/")[1].split("/")[1]
                yield f"mailabs/{m.name}", "mailabs", "train", spk, t.extractfile(m).read()


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


def iter_indiefake(tp):
    """Capped, speaker-split IndieFake subset (pack_indiefake manifest): genuine train/test, fakes train/test."""
    with tarfile.open(tp) as t:
        man = {m["member"]: m for m in json.loads(t.extractfile("indiefake/manifest.json").read())}
        for m in t:
            r = man.get(m.name)
            if r:
                yield r, t.extractfile(m).read()


def _crop_job(args):
    uid, raw = args
    try:
        return uid, crop(raw), None
    except Exception as e:
        return uid, None, f"error:{type(e).__name__}"


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


def prep(raw, work, mailabs_tgz, workers, hindi_root=None):
    from multiprocessing import Pool

    shards = work / "shards"
    excl = Counter()
    writers = {s: ShardWriter(shards, s) for s in ("train", "dev", "cal", "test")}
    rows = mlaad_rows(raw)
    print(f"mlaad rows {len(rows)} {Counter((r['split'], r['family'] in TARGETS) for r in rows)}", flush=True)
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
            for uid, x, err in pool.imap_unordered(_crop_job, jobs, chunksize=8):
                r = byuid[uid]
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
                for uid, corpus, split, spk, b in it:
                    cap = GEN_CAP_SPK if split == "train" else EVAL_CAP_SPK
                    if split == "train" and h(f"{corpus}/{spk}") % 10 == 0:
                        split = "dev"
                    k = (corpus, split, spk)
                    if per_spk[k] >= cap or (split in ("train", "dev") and per_corpus[(corpus, split)] >= GEN_CAP_CORPUS):
                        continue
                    per_spk[k] += 1
                    per_corpus[(corpus, split)] += 1
                    meta[uid] = dict(uid=uid, split=split, label=0, family="genuine", source=corpus,
                                     speaker=f"{corpus}/{spk}", parent=uid, locator="")
                    yield uid, b

        meta = {}
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

        for uid, x, err in pool.imap_unordered(_crop_job, admit_all(), chunksize=8):
            r = meta[uid]
            if x is None:
                excl[f"{r['split']}|genuine/{r['source']}|{err or 'short'}"] += 1
                continue
            writers[r["split"]].add(r, x)
    for w in writers.values():
        w.flush()
    man = load_manifest(work)
    counts = Counter((r["split"], r["family"] if r["label"] else "genuine/" + r["source"]) for r in man)
    spk = defaultdict(set)
    for r in man:
        spk[(r["split"], r["family"] if r["label"] else "genuine/" + r["source"])].add(r["speaker"])
    rep = {"made": now(), "counts": {f"{a}|{b}": n for (a, b), n in sorted(counts.items())},
           "speakers_or_voices": {f"{a}|{b}": len(v) for (a, b), v in sorted(spk.items())},
           "excluded": dict(sorted(excl.items())), "checks": leakage(man)}
    (work / "prep.json").write_text(json.dumps(rep, indent=1))
    return rep


def load_manifest(work, splits=None):
    rows = []
    for p in sorted((work / "shards").glob("*.jsonl")):
        if splits and p.name.split("-")[0] not in splits:
            continue
        rows += [json.loads(l) for l in p.read_text().splitlines()]
    return rows


def leakage(man):
    def sets(split, key, label=None):
        return {r[key] for r in man if r["split"] == split and (label is None or r["label"] == label)}

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
        p = self.work / "shards" / r["shard"]
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


def train(work, drive, encoder, steps, batch, workers, lr_enc, lr_head, eval_every, tag="c2", freeze_below=0):
    import torch
    from torch.utils.data import DataLoader, IterableDataset

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
            groups[("h" if hindi else ("t" if tgt else "d"), r["family"])].append(i)
    kinds = {"g": 0.5, "t": 0.28, "d": 0.14, "h": 0.08}
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
            if best is None or m["key"] > best["key"]:
                best = {"step": step + 1, **m}
                torch.save(model.state_dict(), ck / "best.pt")
            model.train()
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step + 1, "best": best, "hist": hist}, ck / "last.tmp")
            os.replace(ck / "last.tmp", ck / "last.pt")
            (work / f"train-{tag}.json").write_text(json.dumps({"tag": tag, "encoder": encoder, "best": best,
                                                                "hist": hist}, indent=1))
            mirror(work, drive, [f"train-{tag}.json", f"ckpt-{tag}/best.pt", f"ckpt-{tag}/last.pt"])
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


def dev_metric(rows, sc):
    c = cutoff_at([s for r, s in zip(rows, sc) if r["label"] == 0])
    det = {}
    for f in TARGETS:
        v = [s >= c for r, s in zip(rows, sc) if r["family"] == f]
        det[f] = round(sum(v) / len(v), 4) if v else None
    allf = [s >= c for r, s in zip(rows, sc) if r["label"] == 1]
    return {"cutoff": c, "det": det, "pooled": round(sum(allf) / len(allf), 4),
            "key": min(v for v in det.values() if v is not None)}


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
        res[split] = (rows, [simple_features(st.x(r)) for r in rows])
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


def signoff(work, protocol, commit):
    p = work / "signoff.json"
    if p.exists():
        raise SystemExit("signoff exists (write-once)")
    if (work / "eval-scores.jsonl").exists():
        raise ValueError("test already scored")
    test = sorted(load_manifest(work, {"test"}), key=lambda r: r["uid"])
    tm = hashlib.sha256("\n".join(f"{r['uid']}|{r['label']}|{r['family']}|{r['speaker']}" for r in test).encode()).hexdigest()
    cut = json.loads((work / "cutoff.json").read_text())
    if cut["model_sha256"] != fsha(work / "selection.json"):
        raise ValueError("model changed after calibration")
    rec = {"signed": now(), "protocol_sha256": fsha(protocol), "code_commit": commit, "code_sha256": fsha(__file__),
           "model_sha256": cut["model_sha256"], "cutoff": cut["cutoff"], "test_manifest_sha256": tm,
           "test_rows": dict(Counter(r["family"] if r["label"] else "genuine/" + r["source"] for r in test))}
    p.write_text(json.dumps(rec, indent=1))
    return rec


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


def evaluate(work, encoder=None):
    s = json.loads((work / "signoff.json").read_text())
    if fsha(work / "selection.json") != s["model_sha256"]:
        raise ValueError("model changed after signoff")
    st = Store(work, {"test"})
    rows = sorted(st.rows, key=lambda r: r["uid"])
    tm = hashlib.sha256("\n".join(f"{r['uid']}|{r['label']}|{r['family']}|{r['speaker']}" for r in rows).encode()).hexdigest()
    if tm != s["test_manifest_sha256"]:
        raise ValueError("test manifest changed after signoff")
    scorer = Scorer(work)
    c = s["cutoff"]
    res = {"signoff_sha256": fsha(work / "signoff.json"), "evaluated": now(), "cutoff": c, "channels": {}}
    with (work / "eval-scores.jsonl").open("w") as f:
        for ch in CHANNELS_EVAL:
            sc = scorer(st, rows, ch)
            for r, v in zip(rows, sc):
                f.write(json.dumps({"uid": r["uid"], "channel": ch, "score": v}) + "\n")
            cells = defaultdict(lambda: defaultdict(list))
            for r, v in zip(rows, sc):
                hindi_gen = not r["label"] and r["source"] in HINDI_SOURCES
                cells[r["family"] if r["label"] else ("genuine_hindi" if hindi_gen else "genuine")][r["speaker"]].append(int(v >= c))
                if not r["label"]:
                    cells["genuine/" + r["source"]][r["speaker"]].append(int(v >= c))
                    if r["source"] in ("librittsr", "vctk", "globe"):
                        cells["genuine_read_only"][r["speaker"]].append(int(v >= c))
            t = {}
            for k, cl in sorted(cells.items()):
                flat = [x for v in cl.values() for x in v]
                t[k] = {"flagged": sum(flat), "n": len(flat), "rate": round(sum(flat) / len(flat), 4),
                        "wilson95": wilson(sum(flat), len(flat)), "clusters": len(cl), "cluster_boot95": boot(cl)}
            g = t["genuine"]
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
            verdict = "INSUFFICIENT: untested " + ",".join(untested) if untested else (
                "MISS" if gaps else ("PASS" if lo_ok else "PASS at point estimate; not robust"))
            res["channels"][ch] = {"cells": t, "gaps_in_recordings": gaps, "verdict": verdict}
            print(ch, verdict, {k: f"{v['flagged']}/{v['n']}" for k, v in t.items()}, flush=True)
    (work / "result.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["prep", "audit", "train", "select", "calibrate", "signoff", "evaluate"])
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
        r = prep(a.raw, a.work, a.mailabs, a.workers, a.hindi)
    elif a.phase == "audit":
        r = audit(a.work)
    elif a.phase == "train":
        r = train(a.work, a.drive, a.encoder, a.steps, a.batch, a.workers, a.lr_enc, a.lr_head, a.eval_every,
                  a.tag, a.freeze_below)
    elif a.phase == "select":
        r = select(a.work, [tuple(c.split("=", 1)) for c in a.cands])
    elif a.phase == "calibrate":
        r = calibrate(a.work, a.encoder)
    elif a.phase == "signoff":
        r = signoff(a.work, a.protocol, a.commit)
    else:
        r = evaluate(a.work, a.encoder)
    print(json.dumps(r, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
