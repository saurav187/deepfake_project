"""Bounded SpeechFake slice (inclusionAI/SPEECHFAKE, ModelScope) for the modern intake.

Published *training* partition only (metadata/experiments/baseline/train_en.csv). English VCTK:
  VC_GPTSoVITS, VC_OpenVoice  non-vocoder VC: BD/<sys>/VCTK/<target>_<prompt>/<src utt>.wav
                genuine parent = Real VCTK <src utt> (same words); target identity = <target>
  TTS_CosyVoice modern TTS, stock voices en_female/en_male, VCTK texts:
                BD/CosyVoice/<voice>/VCTK/<utt>.wav; genuine parent = Real VCTK <utt> (same text)
Fish (TTS/VC_FishSpeech) stays isolated (reserved family). Speakers excluded on either side:
the ten PhonemeDF voices and the intake's reserved test fold. Members are range-read from the
split BD ZIP and Real/VCTK-Corpus.zip; nothing else is transferred. Same cache recipe and
output root as modern_intake (source="speechfake").

  python -m pipeline_v3.speechfake_slice --out OUT --meta metadata.zip --per-system 400
"""

import argparse
import csv
import hashlib
import io
import json
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from .modern_intake import PHONEMEDF_REFS, TEST_FOLD, IntakeCache, _treat, fold_of
from .remote_zip import RemoteZip

REVISION = "e315a544246df765336bab9595ed552520444c4a"
META_SHA256 = "3e71dec782599b172d5181322b1a30e58c68f862b24b5f686987625ebd4a78d8"
URL = ("https://www.modelscope.cn/api/v1/datasets/inclusionAI/SPEECHFAKE/repo?Source=SDK"
       f"&Revision={REVISION}&FilePath=")
BD_PARTS = [("BD/BD.z0%d" % i, 42949672960) for i in range(1, 7)] + [("BD/BD.zip", 17754620592)]
REAL_VCTK = [("Real/VCTK-Corpus.zip", 2498726226)]
SOURCE = "speechfake"
SYSTEMS = {"VC_GPTSoVITS": "GPT-SoVITS VC (non-vocoder VC)",
           "VC_OpenVoice": "OpenVoice VC (tone-colour converter; related to RVCBench OpenVoice V2)",
           "TTS_CosyVoice": "CosyVoice TTS (version unverified)"}


def rows(z, name):
    return list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8")))


def parse(system, path):
    """-> (target_speaker, genuine_utt, prompt) or None."""
    p = path.split("/")
    if system.startswith("VC_") and len(p) == 5 and p[2] == "VCTK":
        tgt, utt = p[3].split("_")[0], Path(p[4]).stem
        return tgt, utt, p[3]
    if system == "TTS_CosyVoice" and len(p) == 5 and p[3] == "VCTK":
        return "stock:" + p[2], Path(p[4]).stem, None
    return None


def blocked(spk):
    return spk in PHONEMEDF_REFS or fold_of("VCTK", spk) == TEST_FOLD


def select(meta, per_system):
    z = zipfile.ZipFile(meta)
    train = {r["file"] for r in rows(z, "metadata/experiments/baseline/train_en.csv")}
    real = {r["file"] for r in rows(z, "metadata/Real/VCTK.csv")}
    out, stats = [], {}
    for system in SYSTEMS:
        c = Counter()
        by = defaultdict(list)
        for r in rows(z, f"metadata/BD/{system}.csv"):
            if r["language"] != "en":
                continue
            d = parse(system, r["file"])
            if d is None:
                c["not_vctk_or_unparsed"] += 1
                continue
            tgt, utt, prompt = d
            src = utt.split("_")[0]
            gpath = f"Real/VCTK-Corpus/wav/{src}/{utt}.wav"
            if r["file"] not in train:
                c["fake_not_in_published_train"] += 1
            elif gpath not in real or gpath not in train:
                c["genuine_not_in_published_train"] += 1
            elif blocked(src) or (not tgt.startswith("stock:") and blocked(tgt)):
                c["reserved_speaker"] += 1
            else:
                c["admissible"] += 1
                by[tgt].append(dict(system=system, fake=r["file"], genuine=gpath, target=tgt,
                                    source_speaker=src, utt=utt, prompt=prompt))
        # Round-robin over target identities (stock voices for TTS), hash-ordered: bounded,
        # balanced, reproducible.
        queues = [sorted(v, key=lambda x: hashlib.sha256(x["fake"].encode()).hexdigest())
                  for _, v in sorted(by.items())]
        picked, k = [], 0
        while len(picked) < per_system and any(queues):
            q = queues[k % len(queues)]
            if q:
                picked.append(q.pop(0))
            k += 1
        stats[system] = dict(c, picked=len(picked), targets=len(by))
        out += picked
    return out, stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--per-system", type=int, default=400)
    p.add_argument("--scratch", default="/content/intake-scratch")
    p.add_argument("--streams", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--shard-size", type=int, default=200)
    a = p.parse_args()
    log = lambda m: print(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + m, flush=True)
    if hashlib.sha256(a.meta.read_bytes()).hexdigest() != META_SHA256:
        raise ValueError("SpeechFake metadata.zip differs from the pinned release")
    lock = a.out / "writer.lock"
    if lock.exists() and Path("/proc/" + lock.read_text().strip()).exists():
        raise SystemExit(f"Another cache writer holds {lock}")
    import os
    lock.write_text(str(os.getpid()))
    picked, stats = select(a.meta, a.per_system)
    log(f"SELECT {json.dumps(stats)}")
    cache = IntakeCache(a.out)
    bd = RemoteZip([(URL + n, s) for n, s in BD_PARTS])
    rv = RemoteZip([(URL + n, s) for n, s in REAL_VCTK])
    fakes = {x["fake"] for x in picked}
    gens = {x["genuine"] for x in picked}
    txts = {g.replace("/wav/", "/txt/").replace(".wav", ".txt") for g in gens}
    t0 = time.time()
    # Archive member names may or may not carry the leading "BD/" / "Real/" component.
    alt = {f.removeprefix("BD/"): f for f in fakes}
    bdir = {alt[n.removeprefix("BD/")]: v for n, v in
            bd.directory(lambda n: n.removeprefix("BD/") in alt).items()}
    rset = {g.removeprefix("Real/") for g in gens | txts}
    rdir_raw = rv.directory(lambda n: n.removeprefix("Real/") in rset)
    rdir = {"Real/" + n.removeprefix("Real/"): v for n, v in rdir_raw.items()}
    log(f"DIRECTORY bd={len(bdir)}/{len(fakes)} real={len(rdir)}/{len(gens) + len(txts)} "
        f"{time.time() - t0:.0f}s (bd entries {bd.n_entries}, real entries {rv.n_entries})")
    texts = {}
    for t in txts:
        if t in rdir:
            texts[t] = rv.read(rdir[t]).decode("utf-8", "replace").strip()
    objects = [(f, bd, bdir.get(f)) for f in sorted(fakes)] + [(g, rv, rdir.get(g)) for g in sorted(gens)]
    todo = [o for o in objects if (SOURCE, o[0]) not in cache]

    def get(o):
        name, z, info = o
        if info is None:
            return name, None, "member_not_in_archive_directory", None
        try:
            raw = z.read(info)
            return name, raw, None, info
        except Exception as e:
            return name, None, f"download_error:{type(e).__name__}", info

    with ThreadPoolExecutor(a.streams) as net, ProcessPoolExecutor(a.workers) as cpu:
        for i in range(0, len(todo), a.shard_size):
            got = list(net.map(get, todo[i:i + a.shard_size]))
            ok = [g for g in got if g[1] is not None]
            treated = list(cpu.map(_treat, [g[1] for g in ok], chunksize=4))
            records = []
            for (name, raw, err, info), (b, reason, dur) in zip(ok, treated):
                part = (BD_PARTS if name.startswith("BD/") else REAL_VCTK)[info[0]][0]
                loc = {"repo": "modelscope:inclusionAI/SPEECHFAKE", "revision": REVISION, "path": name,
                       "archive_part": part, "zip_crc32": info[5], "source_sha256": hashlib.sha256(raw).hexdigest(),
                       "bytes": len(raw), "source_duration_s": dur}
                records.append(((SOURCE, name, loc), b, reason))
            for name, raw, err, info in got:
                if raw is None and err == "member_not_in_archive_directory":
                    records.append(((SOURCE, name, {"path": name}), None, err))
            bad = [g for g in got if g[2] and g[2].startswith("download_error")]
            if bad:
                log(f"RETRY-LATER {len(bad)}")
            if records:
                name = cache._publish(records, a.scratch)
                log(f"SHARD {name} +{len(records)} eligible={sum(r[1] is not None for r in records)} "
                    f"({i + len(got)}/{len(todo)}) {time.time() - t0:.0f}s")
    out = a.out / "speechfake"
    out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for x in picked:
        fe, ge = cache.entries.get((SOURCE, x["fake"])), cache.entries.get((SOURCE, x["genuine"]))
        t = texts.get(x["genuine"].replace("/wav/", "/txt/").replace(".wav", ".txt"))
        pairs.append({**x, "family": SYSTEMS[x["system"]], "speaker": "VCTK/" + x["target"],
                      "parent_speaker": "VCTK/" + x["source_speaker"], "text": t,
                      "published_partition": "baseline/train_en",
                      "fold": fold_of("VCTK", x["source_speaker"]),
                      "fake_eligible": bool(fe and fe["eligible"]), "fake_reason": fe and fe["reason"],
                      "genuine_eligible": bool(ge and ge["eligible"]), "genuine_reason": ge and ge["reason"],
                      "pair_eligible": bool(fe and ge and fe["eligible"] and ge["eligible"]),
                      "use": "train_candidate", "research_only": False,
                      "licence": "CC BY 4.0 (SpeechFake); VCTK CC BY 4.0"})
    (out / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pairs))
    cov = defaultdict(lambda: Counter())
    spk = defaultdict(set)
    for r in pairs:
        cov[r["system"]]["pairs"] += 1
        cov[r["system"]]["eligible_pairs"] += r["pair_eligible"]
        if r["pair_eligible"]:
            spk[r["system"], "target"].add(r["target"])
            spk[r["system"], "parent"].add(r["source_speaker"])
            spk[r["system"], "genuine"].add(r["genuine"])
    report = {"source": {"repo": "modelscope:inclusionAI/SPEECHFAKE", "revision": REVISION,
                         "metadata_sha256": META_SHA256, "licence": "CC BY 4.0 with GPL/MPL notices for "
                         "ChatTTS, EdgeTTS, Seed-VC, FireRedTTS (none selected)"},
              "selection": stats,
              "coverage": {s: {**c, **{k2: len(v) for (s2, k2), v in spk.items() if s2 == s}} for s, c in cov.items()},
              "texts_resolved": sum(r["text"] is not None for r in pairs),
              "excluded_speakers": {"phonemedf": sorted(PHONEMEDF_REFS), "test_fold": TEST_FOLD},
              "notes": ["Fish (TTS/VC_FishSpeech) not fetched: reserved held-out family.",
                        "CosyVoice TTS uses two stock voices: genuine parent shares text, not identity.",
                        "Fold follows the genuine parent's (source) speaker; channel siblings inherit it."],
              "seconds": time.time() - t0}
    (out / "lineage.json").write_text(json.dumps(report, indent=1))
    log(f"DONE {json.dumps(report['coverage'])}")
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
