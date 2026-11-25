"""V2-L: separately named LibriSeVoc-only English baseline (owner approval 25 Nov 2026).

ASVspoof 5 A08 is excluded because the public release has no transcripts or verified MLS
reader mapping. Everything else follows V2-A (pipeline_v3/v2a.py): the same four vocoder
families (MelGAN, Parallel WaveGAN, WaveGrad, WaveRNN), V1 reader ownership, text
exclusion, frozen 4 s crop + RMS normalisation (v2a.treated), complete four-vocoder parents,
the 0.60 gate on every panel, and V1's calibrate/threshold/validate and Chatterbox select
folds for comparison.

Extraction uses bounded CPU parallelism. Before the full run it checks (1) byte identity
of a sample against the serial V2-A preprocessing and (2) throughput on 1,000 eligible clips
including ZIP reads and scratch writes; below --min-rate clips/s it stops and reports the
bottleneck instead of committing the VM to a long run.

  python -m pipeline_v3.v2l prepare --data $D --scratch /content --workers 8
"""

import argparse
import hashlib
import re
import json
import os
import shutil
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import data
from .match import prompt, short
from .v2a import FAMILIES, LSV, V1, aucs, balance, read_csv, save, treated, write_catalog

VERSION = "v2l-lsv-norm-1"
POLICY = "v2l-parent-controls-v1"
ZIP = "development/librisevoc/LibriSeVoc.zip"
PANELS = (LSV, *FAMILIES)
IDENTITY_SAMPLE = 40


def text_key(transcript):
    """LibriSpeech-comparable key: casefold, curly apostrophes -> ', other punctuation -> space.
    (match.prompt alone keeps punctuation, so LibriTTS text would never match V1's
    unpunctuated LibriSpeech transcripts.)"""
    t = transcript.replace("\u2019", "'").replace("\u2018", "'")
    return prompt(re.sub(r"[^\w\s']", " ", t))
BENCH = 1000


def candidates(root, work):
    from .v2a import lineage

    selection = read_csv(root / "stage0-en-ecapa-xtts/selection.csv")
    owners = {r["speaker"]: r["split"] for r in selection if r["label"] == "0"}
    texts = defaultdict(set)
    for r in selection:
        texts[r["text"]].add(r["split"])
    evidence = lineage(work / "lineage.csv")
    if any(src != LSV for src, _ in evidence):
        raise ValueError("V2-L lineage must contain LibriSeVoc rows only")
    groups = defaultdict(list)
    for r in read_csv(root / "v2/librisevoc/manifest.csv"):
        uid = r["parent"].split("/")[-1]
        if (LSV, uid) in evidence and r["generator"] in ("genuine", *FAMILIES):
            e = evidence[LSV, uid]
            if e["speaker"] != r["speaker"]:
                raise ValueError("LibriSeVoc canonical speaker mismatch")
            groups[uid].append((r["member"], r["generator"], e))
    planned, excluded = [], Counter()
    for uid, members in sorted(groups.items()):
        if {g for _, g, _ in members} != {"genuine", *FAMILIES}:
            raise ValueError(f"Incomplete four-vocoder parent {uid}")
        sp = members[0][2]["speaker"]
        split = owners.get(sp)
        if split not in {"train", "select"}:
            excluded["reader_owned_by_v1_" + str(split)] += 1
            continue
        key = text_key(members[0][2]["transcript"])
        if texts[key] - {split}:
            excluded["text_in_other_v1_fold"] += 1
            continue
        for member, system, e in members:
            fake = system != "genuine"
            rid = short(f"{LSV}/{member}")
            planned.append({
                "id": rid, "member": member, "source": LSV, "pair": LSV, "speaker": sp,
                "split": split, "label": str(int(fake)), "voice": sp if fake else "-",
                "parent": f"{LSV}/{uid}", "text": key, "language": "en", "channel": "clean",
                "generator": system if fake else "-", "domain": "seen" if fake else "genuine",
                "prepared_view": "v2a-fixed-norm", "path": f"audio/{rid}.wav",
            })
    by_text = defaultdict(set)
    for r in planned:
        by_text[r["text"]].add(r["split"])
    kept = [r for r in planned if len(by_text[r["text"]]) == 1]
    excluded["text_split_conflict_rows"] = len(planned) - len(kept)
    return kept, dict(excluded)


_Z = {}


def _work(args):
    zpath, dev, r = args
    from .audit import descriptors

    t0 = time.perf_counter()
    if zpath not in _Z:
        _Z[zpath] = zipfile.ZipFile(zpath)
    raw = _Z[zpath].read(r["member"])
    t1 = time.perf_counter()
    b = treated(raw)
    t2 = time.perf_counter()
    if b is None:
        return r["id"], None, None, (t1 - t0, t2 - t1, 0.0, 0.0)
    path = Path(dev) / r["path"]
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(b)
    os.replace(tmp, path)
    t3 = time.perf_counter()
    f = descriptors(b)
    t4 = time.perf_counter()
    return r["id"], hashlib.sha256(b).hexdigest(), f, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)


def extract(zpath, dev, rows, workers, log):
    out, timing = {}, []
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, (rid, sha, f, t) in enumerate(
            ex.map(_work, [(str(zpath), str(dev), r) for r in rows], chunksize=4), 1
        ):
            out[rid] = (sha, f)
            timing.append(t)
            if i % 1000 == 0 or i == len(rows):
                rate = i / (time.time() - t0)
                log(f"PROGRESS {i}/{len(rows)} clips {rate:.2f}/s eta {(len(rows) - i) / rate / 60:.1f} min")
    return out, timing, time.time() - t0


def bottleneck(timing, wall, workers):
    n = len(timing)
    parts = dict(zip(("zip_read", "crop_norm", "write", "descriptors"),
                     (sum(t[k] for t in timing) / n for k in range(4))))
    cpu_bound_rate = workers / max(sum(parts.values()), 1e-9)
    return {"mean_seconds_per_clip": parts, "dominant": max(parts, key=parts.get),
            "per_worker_limit_clips_per_s": cpu_bound_rate, "measured_clips_per_s": n / wall}


def panels(rows, features):
    from .en_treatment import score

    by_parent = defaultdict(list)
    for r in rows:
        if r["label"] == "1":
            by_parent[r["parent"]].append(r)
    rotated = {min(g, key=lambda r: short(VERSION + r["id"]))["id"] for g in by_parent.values()}
    result = {}
    for name in PANELS:
        panel = [r for r in rows if name == LSV or r["generator"] in {name, "-"}]
        if name == LSV:
            panel = [r for r in panel if r["label"] == "0" or r["id"] in rotated]
        worst, values = score([aucs(panel, [features[r["id"]] for r in panel])])
        result[name] = {"worst": worst, "oriented_auc": values,
                        "counts": dict(Counter(r["split"] + "/" + r["label"] for r in panel))}
    return result


def prepare(root, work, scratch, workers, min_rate, log=print):
    rows, excluded = candidates(root, work)
    preview = balance(rows)
    data.check(preview)
    coverage = {s: {"readers": len({r["speaker"] for r in preview if r["split"] == s}),
                    "parents": len({r["parent"] for r in preview if r["split"] == s}),
                    "clips": sum(r["split"] == s for r in preview)} for s in ("train", "select")}
    save(work / "coverage.json", {"coverage": coverage, "excluded_parents": excluded})
    log(f"coverage {json.dumps(coverage)} excluded {json.dumps(excluded)}")
    for s, need in (("train", 20), ("select", 5)):
        if coverage[s]["readers"] < need:
            raise ValueError(f"Insufficient LibriSeVoc {s} reader coverage")
    recipe = {"version": VERSION, "treatment": "norm", "families": list(FAMILIES),
              "selection_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
              "lineage_sha256": data.digest(work / "lineage.csv"), "threshold": 0.60,
              "excluded_source": "asvspoof5 A08 (no public transcripts or verified MLS reader mapping)"}
    frozen = work / "recipe.json"
    if frozen.exists() and json.loads(frozen.read_text()) != recipe:
        raise ValueError("Frozen V2-L selection changed")
    save(frozen, recipe)
    dev = scratch / "development"
    (dev / "audio").mkdir(parents=True)
    zpath = root / ZIP
    if shutil.disk_usage(scratch).free < len(rows) * 140_000 + 20e9:
        raise ValueError("Scratch space too small for treated crops")
    from .clip_cache import ClipCache

    cache = ClipCache(root)
    order = sorted(rows, key=lambda r: short(VERSION + "/order/" + r["id"]))
    items = [(r["member"], r["member"]) for r in order]  # source_id = publisher ZIP member
    # (1) byte identity: shared cache output vs the serial V2-A preprocessing (v2a.treated)
    sample = items[:IDENTITY_SAMPLE]
    cache.ensure_zip(LSV, zpath, sample, scratch, workers, shard_size=IDENTITY_SAMPLE, log=log)
    with zipfile.ZipFile(zpath) as z:
        mismatch = [m for m, _ in sample
                    if (lambda b, c: (b is None) != (c is None) or (b is not None and b != c))(
                        treated(z.read(m)), cache.get(LSV, m))]
    identity = {"sample": len(sample), "mismatches": len(mismatch), "identical": not mismatch}
    log(f"identity {json.dumps(identity)}")
    if mismatch:
        save(work / "benchmark.json", {"identity": identity})
        raise ValueError("Cached output differs from serial V2-A preprocessing")
    # (2) throughput on 1,000 clips including ZIP reads, treatment and Drive shard writes
    bench_items = items[IDENTITY_SAMPLE:IDENTITY_SAMPLE + BENCH]
    st = cache.ensure_zip(LSV, zpath, bench_items, scratch, workers, shard_size=BENCH, log=log)
    n = max(st["processed"], 1)
    report = {"identity": identity, "workers": workers, "cpus": os.cpu_count(),
              "clips": st["processed"], "already_cached": st["skipped_cached"],
              "wall_seconds": st["seconds"],
              "clips_per_s": st["processed"] / st["seconds"] if st["processed"] else None,
              "bottleneck": {"mean_zip_read_s": st["read_s"] / n, "mean_treat_s": st["treat_s"] / n,
                             "publish_s_per_clip": st["publish_s"] / n,
                             "worker_limit_clips_per_s": workers / max((st["read_s"] + st["treat_s"]) / n, 1e-9)},
              "min_rate": min_rate, "remaining_clips": sum((LSV, m) not in cache for m, _ in items)}
    if report["clips_per_s"]:
        b = report["bottleneck"]
        b["dominant"] = max(("zip_read", b["mean_zip_read_s"] / workers), ("treat", b["mean_treat_s"] / workers),
                            ("drive_publish", b["publish_s_per_clip"]), key=lambda t: t[1])[0]
        report["eta_minutes_remaining"] = report["remaining_clips"] / report["clips_per_s"] / 60
    # (3) resume check: re-requesting completed clips must process none
    again = cache.ensure_zip(LSV, zpath, items[:IDENTITY_SAMPLE + BENCH], scratch, workers, log=log)
    report["resume_check"] = {"requested": again["requested"], "skipped": again["skipped_cached"],
                              "reprocessed": again["processed"], "ok": again["processed"] == 0}
    save(work / "benchmark.json", report)
    log(f"benchmark {json.dumps(report)}")
    if not report["resume_check"]["ok"]:
        raise ValueError("Cache resume reprocessed completed clips")
    if report["clips_per_s"] is not None and report["clips_per_s"] < min_rate:
        raise SystemExit(f"BENCHMARK_BELOW_TARGET {report['clips_per_s']:.2f} < {min_rate} clips/s; "
                         f"dominant {report['bottleneck']['dominant']}")
    cache.ensure_zip(LSV, zpath, items, scratch, workers, log=log)
    # Materialise this experiment's rows from the shared cache and compute audit descriptors.
    from .audit import descriptors

    features = {}
    blobs = []
    for r in rows:
        b = cache.get(LSV, r["member"])
        if b is not None:
            (dev / r["path"]).write_bytes(b)
            r["sha256"] = hashlib.sha256(b).hexdigest()
            blobs.append((r["id"], b))
    with ProcessPoolExecutor(workers) as ex:
        for (rid, _), f in zip(blobs, ex.map(descriptors, [b for _, b in blobs], chunksize=16)):
            features[rid] = f
    rows = balance([r for r in rows if r["id"] in features])
    data.check(rows)
    for s, need in (("train", 20), ("select", 5)):
        if len({r["speaker"] for r in rows if r["split"] == s}) < need:
            raise ValueError(f"Insufficient LibriSeVoc {s} speaker coverage after cropping")
    result = panels(rows, features)
    passed = all(v["worst"] <= 0.60 for v in result.values())
    audit = dict(version=VERSION, policy=POLICY, languages_audited=["en"], panels=result,
                 lineage_sha256=data.digest(work / "lineage.csv"),
                 matching={LSV: "exact utterance, complete four-vocoder parents"}, **{"pass": passed})
    audit["recipe_sha256"] = data.digest(frozen)
    audit["audited_rows_sha256"] = hashlib.sha256(
        json.dumps(sorted((r["id"], r["split"], r["sha256"]) for r in rows)).encode()).hexdigest()
    previous = work / "mixture-audit.json"
    if previous.exists() and json.loads(previous.read_text()) != audit:
        raise ValueError("Frozen audit inputs or results changed")
    save(previous, audit)
    log(f"audit pass={passed} " + json.dumps({k: round(v['worst'], 3) for k, v in result.items()}))
    if not passed:
        raise ValueError("Frozen V2-L audit failed. See mixture-audit.json")
    rows = [r for r in rows if r["split"] == "train"]
    old = root / V1 / "packed"
    sums = json.loads((old / "checksums.json").read_text())
    if data.digest(old / "index.json") != sums["index.json"]:
        raise ValueError("V1 reference index checksum mismatch")
    transfer = data.entries(old, data.SPLITS - {"train"})
    select_parents = {r["parent"] for r in transfer
                      if r["split"] == "select" and r["generator"] == "chatterboxtts"}
    for r in transfer:
        if r["split"] == "select" and (r["parent"] not in select_parents
                                       or r["label"] == "1" and r["generator"] != "chatterboxtts"):
            continue
        b = data.payload(old, r)
        r = {k: r[k] for k in data.FIELDS + ["sha256"]}
        r["prepared_view"] = "v2a-fixed-norm"
        r["path"] = f"audio/v1-{r['id']}.wav"
        if r["label"] == "1":
            r["domain"] = "unseen"
        (dev / r["path"]).write_bytes(b)
        rows.append(r)
    data.check(rows)
    write_catalog(rows, dev / "catalog.csv")
    target = work / "packed"
    if target.exists():
        index = json.loads((target / "index.json").read_text())
        expected = {r["id"]: {k: r[k] for k in data.FIELDS + ["sha256"]} for r in rows}
        actual = {r["id"]: {k: r[k] for k in data.FIELDS + ["sha256"]} for r in index}
        if actual != expected:
            raise ValueError("Partial publication has different content")
    else:
        data.pack(scratch, "catalog.csv", target)
    audit["index_sha256"] = data.digest(target / "index.json")
    save(work / "audit.json", audit)
    log("PACK_DONE")


def verify_published(work, report, index):
    mixture = json.loads((work / "mixture-audit.json").read_text())
    if (report.get("version") != VERSION or report.get("policy") != POLICY
            or report.get("pass") is not True or report.get("languages_audited") != ["en"]
            or set(report["panels"]) != set(PANELS)
            or not all(p["worst"] <= 0.60 for p in report["panels"].values())
            or report != {**mixture, "index_sha256": data.digest(index)}
            or report["recipe_sha256"] != data.digest(work / "recipe.json")
            or report["lineage_sha256"] != data.digest(work / "lineage.csv")):
        raise ValueError("V2-L audit does not authorize this pack")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["prepare"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--scratch", type=Path, default=Path("/content"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min-rate", type=float, default=3.0)
    a = p.parse_args()
    work = a.data / "v2" / VERSION
    work.mkdir(parents=True, exist_ok=True)
    import fcntl

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    with (work / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (work / "audit.json").exists():
            from .cli import bound_audit

            bound_audit(work, work / "packed/index.json")
            log("PACK_DONE (verified existing)")
            return
        if (work / "mixture-audit.json").exists() and not json.loads((work / "mixture-audit.json").read_text())["pass"]:
            raise ValueError("Frozen V2-L audit already failed; retain formal result")
        with tempfile.TemporaryDirectory(dir=a.scratch) as tmp:
            prepare(a.data, work, Path(tmp), max(1, min(a.workers, (os.cpu_count() or 2) - 1)), a.min_rate, log)


if __name__ == "__main__":
    main()
