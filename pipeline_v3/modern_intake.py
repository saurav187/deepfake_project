"""Modern-data intake (college/24_MODERN_DATA_INTAKE.md): RVCBench matched cloning first.

Sole writer of <out> (default layout mirrors MyDrive/audio-v3/development/modern-intake/v1).
Work happens on local scratch; `publish` copies verified files to Drive one stream at a time.

  <out>/clip-cache/fixed4s-rmsnorm-v1/{meta.json,shards/*.tar,manifest/*.jsonl}
      Same recipe and record format as pipeline_v3.clip_cache (fixed 4 s crop + RMS norm,
      code fingerprint checked). `locator` holds byte lineage: repo, revision, path,
      source_sha256 (computed on every downloaded byte), publisher_sha256 (fakes), bytes,
      source duration. Each genuine recording is stored once (keyed by its publisher path).
  <out>/rvcbench/pairs.jsonl      one row per generated clip: pair/parent/speaker/text/prompt
                                   lineage, family, licence, role, fold, eligibility.
  <out>/rvcbench/lineage.json     upstream roles, reservations, duplicates, coverage.
  <out>/rvcbench/diagnostic.json  5-fold speaker-grouped train-only acoustic probe (0.60 gate).

  python -m pipeline_v3.modern_intake fetch --meta rvc.parquet --out OUT --stage core
  python -m pipeline_v3.modern_intake lineage --meta rvc.parquet --out OUT --speakers SPEAKERS.TXT
  python -m pipeline_v3.modern_intake diagnostic --out OUT
  python -m pipeline_v3.modern_intake publish --out OUT --dest DRIVE_DIR

All RVCBench clips are CC BY-NC 4.0 plus per-system terms: research-only, never app assets.
"""

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from .clip_cache import PREP_VERSION, ClipCache, fingerprint

REPO = "Nanboy/RVCBench-Cloned-Speech"
REVISION = "365897b08925e241672731f25bcb5edc5857417f"
META_BYTES = 4477689
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/"
SOURCE = "rvcbench"
CORPORA = ("VCTK", "Libritts")

# Stage -> models. "core" is the first deliverable; "newer" is a disjoint newer system;
# "fish" is the reserved held-out family (evaluation only, test speakers only).
STAGES = {
    "core": ("F5-TTS", "CosyVoice 2", "OpenVoice V2"),
    "newer": ("Qwen3-TTS", "VibeVoice"),
    "fish": ("Fish Audio S2", "OpenAudio S1-mini"),
}
FAMILY = {
    "F5-TTS": dict(family="F5-TTS (flow matching, DiT)", licence="CC BY-NC 4.0",
                   related="also in SCDF; not in V1 train or MLAAD panel"),
    "CosyVoice 2": dict(family="CosyVoice (LM + flow matching)", licence="Apache-2.0",
                        related="SpeechFake lists 'CosyVoice' without a verified version"),
    "OpenVoice V2": dict(family="OpenVoice (tone-colour VC over MeloTTS base)", licence="MIT",
                         related="base TTS is MeloTTS, which is a V1 training family"),
    "Qwen3-TTS": dict(family="Qwen3-TTS (codec LM)", licence="Apache-2.0", related="none known"),
    "VibeVoice": dict(family="VibeVoice (LM + diffusion head)", licence="MIT, research use",
                      related="none known"),
    "Fish Audio S2": dict(family="Fish", licence="Fish Audio Research License (non-commercial)",
                          related="Fish release s2-pro; MLAAD 'FishTTS' exact checkpoint unverified, "
                                  "not assumed identical"),
    "OpenAudio S1-mini": dict(family="Fish", licence="CC BY-NC-SA 4.0",
                              related="Fish release s1-mini; MLAAD 'FishTTS' exact checkpoint "
                                      "unverified, not assumed identical"),
}
# Ten PhonemeDF VCTK reference voices (V1 XTTS targets): reserved across folds.
from .phonemedf import REFS as PHONEMEDF_REFS  # noqa: E402

N_FOLDS = 5
TEST_FOLD = 0  # speaker-level reserved test fold (also the only speakers Fish is scored on)


def fold_of(corpus, speaker):
    h = hashlib.sha256(f"modern-intake-v1/{corpus}/{speaker}".encode()).hexdigest()
    return int(h, 16) % N_FOLDS


class IntakeCache(ClipCache):
    """ClipCache with the same recipe, rooted in the intake directory (own writer)."""

    def __init__(self, out):
        self.root = Path(out) / "clip-cache" / PREP_VERSION
        (self.root / "shards").mkdir(parents=True, exist_ok=True)
        (self.root / "manifest").mkdir(exist_ok=True)
        meta = self.root / "meta.json"
        fp = fingerprint()
        if meta.exists():
            m = json.loads(meta.read_text())
            if m["fingerprint"] != fp or m["version"] != PREP_VERSION:
                raise ValueError("Preprocessing code changed; use a new PREP_VERSION")
        else:
            meta.write_text(json.dumps({"version": PREP_VERSION, "fingerprint": fp,
                                        "treatment": "audio.fixed_crop 4 s centre crop, then "
                                        "en_treatment.apply('norm'): x*0.05/rms, 16-bit WAV"}, indent=2))
        self.version = PREP_VERSION
        self._handles = {}
        self.entries = self._load()


def load_meta(path):
    import pyarrow.parquet as pq

    if os.path.getsize(path) != META_BYTES:
        raise ValueError("metadata.parquet size differs from the pinned release")
    return pq.read_table(path).to_pylist()


def selected(rows, models):
    return [r for r in rows if r["condition"] == "clean" and r["language"] == "en"
            and r["source_dataset"] in CORPORA and r["model"] in models]


def _download(item, tries=4):
    path, expected = item
    err = None
    for k in range(tries):
        try:
            raw = urllib.request.urlopen(BASE + path, timeout=60).read()
            break
        except Exception as e:  # network: bounded retry, then record
            err = e
            time.sleep(2 * (k + 1))
    else:
        return path, None, f"download_error:{type(err).__name__}"
    actual = hashlib.sha256(raw).hexdigest()
    if expected and actual != expected:
        return path, None, "publisher_sha256_mismatch"
    return path, raw, None


def _treat(raw):
    from .audio import SR, decode
    from .clip_cache import treat

    try:
        dur = len(decode(raw)) / SR
        b = treat(raw)
        return b, (None if b is not None else "shorter_than_4s_after_trim"), dur
    except Exception as e:
        return None, f"error:{type(e).__name__}", None


def fetch(a, log):
    rows = selected(load_meta(a.meta), STAGES[a.stage])
    if a.limit:
        keep = sorted({r["pair_id"] for r in rows})[: a.limit]
        rows = [r for r in rows if r["pair_id"] in keep]
    cache = IntakeCache(a.out)
    objects = {}
    for r in rows:
        objects[r["file_name"]] = r["sha256"]
        objects.setdefault(r["ground_truth_file_name"], None)  # genuine stored once
    todo = [(p, h) for p, h in sorted(objects.items()) if (SOURCE, p) not in cache]
    log(f"FETCH stage={a.stage} rows={len(rows)} objects={len(objects)} todo={len(todo)}")
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    t0, done, nbytes = time.time(), 0, 0
    with ThreadPoolExecutor(a.streams) as net, ProcessPoolExecutor(a.workers) as cpu:
        for i in range(0, len(todo), a.shard_size):
            group = todo[i:i + a.shard_size]
            got = list(net.map(_download, group))
            raws = [raw for _, raw, _ in got if raw is not None]
            treated = iter(list(cpu.map(_treat, raws, chunksize=4)))
            records, retry = [], []
            for (path, expected), (_, raw, err) in zip(group, got):
                loc = {"repo": REPO, "revision": REVISION, "path": "data/" + path,
                       "publisher_sha256": expected}
                if raw is None:
                    retry.append((path, err))  # not recorded: a resumed run retries it
                    continue
                b, reason, dur = next(treated)
                loc.update(source_sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                           source_duration_s=dur)
                nbytes += len(raw)
                records.append(((SOURCE, path, loc), b, reason))
            if retry:
                log(f"RETRY-LATER {len(retry)}: {Counter(e for _, e in retry)}")
            if not records:
                continue
            name = cache._publish(records, scratch)
            done += len(group)
            el = time.time() - t0
            log(f"SHARD {name} +{len(group)} ({done}/{len(todo)}) eligible="
                f"{sum(b is not None for _, b, _ in records)} {nbytes / 1e6:.1f} MB "
                f"{done / el:.2f} obj/s eta {(len(todo) - done) / max(done / el, 1e-9) / 60:.1f} min")
    log(f"FETCH done stage={a.stage} seconds={time.time() - t0:.0f}")


def librispeech_speakers(path):
    """LibriSpeech SPEAKERS.TXT -> {id: (sex, subset, name)} (LibriTTS shares speaker IDs)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(";") or not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 5:
            out[parts[0]] = (parts[1], parts[2], parts[4])
    return out


# All M-AILABS English readers (en_US + en_UK), a superset of the 3 MLAAD panel readers.
MAILABS_EN_READERS = "elizabeth_klett,judy_bieber,mary_ann,elliot_miller"


def norm_name(n):
    return " ".join(n.replace("_", " ").lower().split())


def upstream_role(corpus, speaker, libri):
    if corpus == "VCTK":
        if speaker in PHONEMEDF_REFS:
            return "reserved:phonemedf_reference_voice(V1 XTTS target)"
        return "vctk:no_official_split"
    sub = libri.get(speaker, (None, None, None))[1]
    if sub is None:
        return "unresolved:speaker_not_in_SPEAKERS.TXT"
    if sub.startswith("train"):
        return "librispeech:" + sub
    # dev-* readers are V1's reader pool (fetch.LIBRI); test-* are upstream evaluation.
    return "upstream_eval:" + sub + (" (V1 reader pool)" if sub.startswith("dev") else "")


def lineage(a, log):
    rows = selected(load_meta(a.meta), {m for s in STAGES.values() for m in s})
    cache = IntakeCache(a.out)
    libri = librispeech_speakers(a.speakers)
    panel = {norm_name(n) for n in (a.panel_readers or MAILABS_EN_READERS).split(",") if n.strip()}
    v1_readers = set(json.loads(Path(a.v1_readers).read_text())) if a.v1_readers else None
    out = Path(a.out) / "rvcbench"
    out.mkdir(parents=True, exist_ok=True)

    def entry(path):
        return cache.entries.get((SOURCE, path))

    pairs, pcm = [], defaultdict(set)
    gt_users = defaultdict(set)
    for e in cache.entries.values():
        if e["eligible"]:
            pcm[e["sha256"]].add(e["source_id"])
    for r in rows:
        fe, ge = entry(r["file_name"]), entry(r["ground_truth_file_name"])
        if fe is None or ge is None:
            continue  # stage not fetched yet
        corpus, spk = r["source_dataset"], r["speaker_id"]
        role = upstream_role(corpus, spk, libri)
        name = norm_name(libri.get(spk, (None, None, ""))[2]) if corpus == "Libritts" else ""
        if v1_readers is not None and corpus == "Libritts" and f"librispeech/{spk}" in v1_readers:
            role = "reserved:V1_reader"
        if name and name in panel:
            role = "reserved:MLAAD_panel_reader"
        fold = fold_of(corpus, spk)
        fish = FAMILY[r["model"]]["family"] == "Fish"
        if role.startswith(("reserved", "upstream_eval", "unresolved")):
            use = "eval_descriptive_only"
        elif fish:
            use = "heldout_family_test" if fold == TEST_FOLD else "excluded:fish_on_train_speaker"
        elif fold == TEST_FOLD:
            use = "reserved_test_speaker"
        else:
            use = "train_candidate"
        gt_users[r["ground_truth_file_name"]].add(r["pair_id"])
        pairs.append({
            "pair_id": r["pair_id"], "clip_id": r["clip_id"], "model": r["model"], **FAMILY[r["model"]],
            "corpus": corpus, "speaker": f"{corpus}/{spk}", "text": r["text"],
            "target_utterance": r["target_utterance"], "prompt_utterance": r["prompt_utterance"],
            "prompt_file": r["prompt_file_name"], "fake": r["file_name"],
            "genuine": r["ground_truth_file_name"], "upstream_role": role, "fold": fold, "use": use,
            "fake_eligible": fe["eligible"], "fake_reason": fe["reason"],
            "genuine_eligible": ge["eligible"], "genuine_reason": ge["reason"],
            "pair_eligible": fe["eligible"] and ge["eligible"],
            "prompt_is_target": r["prompt_utterance"] == r["target_utterance"],
            "research_only": True,
        })
    # PCM duplicates across distinct source objects, and repeated genuine parents.
    dups = {h: sorted(ids) for h, ids in pcm.items() if len(ids) > 1}
    (out / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))

    def cov(sel):
        c = defaultdict(lambda: {"pairs": 0, "eligible_pairs": 0, "speakers": set(), "parents": set()})
        for p in sel:
            k = f"{p['corpus']}|{p['model']}|{p['use']}"
            c[k]["pairs"] += 1
            c[k]["eligible_pairs"] += p["pair_eligible"]
            if p["pair_eligible"]:
                c[k]["speakers"].add(p["speaker"])
                c[k]["parents"].add(p["genuine"])
        return {k: {**v, "speakers": len(v["speakers"]), "parents": len(v["parents"])} for k, v in sorted(c.items())}

    roles = defaultdict(set)
    for p in pairs:
        roles[p["upstream_role"]].add(p["speaker"])
    exclusions = Counter((p["model"], p["fake_reason"] or p["genuine_reason"]) for p in pairs if not p["pair_eligible"])
    report = {
        "source": {"repo": REPO, "revision": REVISION, "metadata_bytes": META_BYTES,
                   "licence": "CC BY-NC 4.0 + per-system terms; research-only; prompts/ground truth CC BY 4.0"},
        "prep_version": PREP_VERSION, "fingerprint": fingerprint(),
        "cache_entries": len(cache.entries), "cache_eligible": sum(e["eligible"] for e in cache.entries.values()),
        "pairs": len(pairs), "eligible_pairs": sum(p["pair_eligible"] for p in pairs),
        "speakers_by_role": {k: sorted(v) for k, v in sorted(roles.items())},
        "coverage": cov(pairs),
        "exclusions": {f"{m}|{r}": n for (m, r), n in sorted(exclusions.items())},
        "genuine_parents": len(gt_users),
        "genuine_reuse_max_pairs": max((len(v) for v in gt_users.values()), default=0),
        "prompt_equals_target": sum(p["prompt_is_target"] for p in pairs),
        "pcm_duplicates": dups,
        "v1_reader_list_checked": v1_readers is not None,
        "panel_readers_checked": sorted(panel),
        "notes": [
            "LibriTTS speakers outside LibriSpeech train-* are upstream dev/test; dev-* is V1's reader pool.",
            "VCTK has no official split; the ten PhonemeDF reference voices are reserved.",
            "VCC 2018 speakers are DAPS-derived (VCC2S*/VCC2T*), not VCTK/LibriTTS: no identity overlap.",
            "Channel siblings (clean/opus/phone) are derived from one cached clip and inherit its fold.",
            f"Speaker fold = sha256('modern-intake-v1/<corpus>/<speaker>') mod {N_FOLDS}; fold {TEST_FOLD} reserved test.",
        ],
    }
    (out / "lineage.json").write_text(json.dumps(report, indent=1))
    log(f"LINEAGE pairs={len(pairs)} eligible={report['eligible_pairs']} parents={len(gt_users)} "
        f"dups={len(dups)} roles={ {k: len(v) for k, v in roles.items()} }")


def _feat(b):
    from .audit import descriptors

    return descriptors(b)


def diagnostic(a, log):
    from .en_treatment import score
    from .v2a import aucs

    out = Path(a.out) / "rvcbench"
    cache = IntakeCache(a.out)
    train = []
    for src, sub, mk in (("rvcbench", "rvcbench", "model"), ("speechfake", "speechfake", "system")):
        f = Path(a.out) / sub / "pairs.jsonl"
        if f.exists():
            for x in f.read_text().splitlines():
                q = json.loads(x)
                if q["use"] == "train_candidate" and q["pair_eligible"]:
                    train.append((src, f"{src}|{q[mk]}", q))
    # Balance: each fake with its genuine parent; a parent is reused at most once per family.
    items, seen = [], set()
    for src, fam, p in sorted(train, key=lambda t: t[2]["fake"]):
        if (fam, p["genuine"]) in seen:
            continue
        seen.add((fam, p["genuine"]))
        g = p.get("parent_speaker", p["speaker"])  # group = genuine parent speaker
        items.append((fam, g, "1", (src, p["fake"])))
        items.append((fam, g, "0", (src, p["genuine"])))
    ids = sorted({i[3] for i in items})
    with ProcessPoolExecutor(a.workers) as ex:
        feats = dict(zip(ids, ex.map(_feat, [cache.get(*i) for i in ids], chunksize=8)))
    speakers = sorted({i[1] for i in items})
    folds = {s: k % N_FOLDS for k, s in enumerate(sorted(speakers, key=lambda s: hashlib.sha256(s.encode()).hexdigest()))}

    def cv(sub):
        per = []
        for k in range(N_FOLDS):
            rr = [{"label": y, "split": "select" if folds[s] == k else "train"} for _, s, y, _ in sub]
            if {r["label"] for r in rr if r["split"] == "select"} != {"0", "1"} or \
               {r["label"] for r in rr if r["split"] == "train"} != {"0", "1"}:
                continue
            per.append(aucs(rr, [feats[i] for *_, i in sub]))
        if not per:
            return {"folds": 0}
        worst, mean = score(per)
        return {"folds": len(per), "worst": worst, "mean_oriented_auc": mean,
                "failed_at_0.60": sorted(k for k, v in mean.items() if v > 0.60),
                "n": len(sub), "speakers": len({s for _, s, _, _ in sub})}

    out.mkdir(parents=True, exist_ok=True)
    result = {"kind": "Modern-intake train-candidate 5-fold speaker-grouped acoustic diagnostic (0.60 gate)",
              "gate": 0.60, "train_speakers": len(speakers),
              "enough_speakers": len(speakers) >= 2 * N_FOLDS,
              "pooled": cv(items),
              "per_family": {m: cv([i for i in items if i[0] == m]) for m in sorted({i[0] for i in items})}}
    result["families_n"] = dict(Counter(i[0] for i in items))
    (Path(a.out) / "diagnostic.json").write_text(json.dumps(result, indent=1))
    log(f"DIAGNOSTIC speakers={len(speakers)} pooled_worst={result['pooled'].get('worst')} "
        f"failed={result['pooled'].get('failed_at_0.60')}")


def publish(a, log):
    """Copy verified local outputs to Drive, one stream, skipping identical files."""
    src, dest = Path(a.out), Path(a.dest)
    n = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.suffix == ".tmp":
            continue
        d = dest / p.relative_to(src)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if d.exists() and hashlib.sha256(d.read_bytes()).hexdigest() == h:
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_name(d.name + ".tmp")
        shutil.copyfile(p, tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != h:
            raise ValueError(f"Copy differs: {p}")
        os.replace(tmp, d)
        n += 1
    log(f"PUBLISH {n} files -> {dest}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["fetch", "lineage", "diagnostic", "publish"])
    p.add_argument("--meta", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stage", choices=sorted(STAGES), default="core")
    p.add_argument("--scratch", default="/content/intake-scratch")
    p.add_argument("--streams", type=int, default=6)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--shard-size", type=int, default=200)
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N pair IDs")
    p.add_argument("--speakers", type=Path)
    p.add_argument("--v1-readers", type=Path, help="JSON list of 'librispeech/<id>' V1 readers")
    p.add_argument("--panel-readers", help="comma-separated M-AILABS reader names")
    p.add_argument("--dest", type=Path)
    a = p.parse_args()
    # Cache writers share writer.lock; report steps each own their outputs.
    lock = a.out / ("writer.lock" if a.cmd in ("fetch", "publish") else f"{a.cmd}.lock")
    a.out.mkdir(parents=True, exist_ok=True)
    logf = (a.out / "intake.log").open("a")

    def log(msg):
        line = time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if lock.exists() and lock.read_text().strip() != str(os.getpid()):
        other = lock.read_text().strip()
        if other and Path(f"/proc/{other}").exists():
            raise SystemExit(f"Another writer holds {lock} (pid {other})")
    lock.write_text(str(os.getpid()))
    try:
        {"fetch": fetch, "lineage": lineage, "diagnostic": diagnostic, "publish": publish}[a.cmd](a, log)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
