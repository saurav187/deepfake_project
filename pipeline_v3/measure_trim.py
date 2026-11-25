"""Measure-only: trimmed speech duration (audio.prepare, the fixed_crop energy trim) of every
verified intake object, then pair eligibility at several thresholds. Writes no cache.

Source bytes are re-fetched, checked against the source_sha256 recorded in the cache
manifest, measured in memory and discarded.

  python -m pipeline_v3.measure_trim objects --out /content/mi/v1 --source rvcbench
  python -m pipeline_v3.measure_trim objects --out /content/mi/v1 --source speechfake
  python -m pipeline_v3.measure_trim report  --out /content/mi/v1
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from .modern_intake import BASE, IntakeCache

THRESHOLDS = (2.0, 3.0, 4.0)


def _trim(raw):
    from .audio import SR, decode, prepare

    try:
        x = decode(raw)
        return len(prepare(x)[0]) / SR, len(x) / SR, None
    except Exception as e:
        return None, None, f"error:{type(e).__name__}"


def objects(a):
    cache = IntakeCache(a.out)
    out = Path(a.out) / "measure"
    out.mkdir(exist_ok=True)
    dest = out / f"trim-{a.source}.jsonl"
    done = {json.loads(l)["source_id"] for l in dest.read_text().splitlines()} if dest.exists() else set()
    ents = [e for (s, _), e in sorted(cache.entries.items()) if s == a.source and e["source_id"] not in done
            and e["locator"].get("source_sha256")]
    if a.source == "rvcbench":
        def fetch(e):
            import urllib.request
            for k in range(4):
                try:
                    return urllib.request.urlopen(BASE + e["source_id"], timeout=60).read()
                except Exception:
                    time.sleep(2 * (k + 1))
            return None
    else:
        from .remote_zip import RemoteZip
        from .speechfake_slice import BD_PARTS, REAL_VCTK, URL

        bd = RemoteZip([(URL + n, s) for n, s in BD_PARTS])
        rv = RemoteZip([(URL + n, s) for n, s in REAL_VCTK])
        want_bd = {e["source_id"].removeprefix("BD/") for e in ents if e["source_id"].startswith("BD/")}
        want_rv = {e["source_id"].removeprefix("Real/") for e in ents if e["source_id"].startswith("Real/")}
        dirs = {"BD/" + n.removeprefix("BD/"): (bd, v) for n, v in bd.directory(lambda n: n.removeprefix("BD/") in want_bd).items()}
        dirs.update({"Real/" + n.removeprefix("Real/"): (rv, v) for n, v in rv.directory(lambda n: n.removeprefix("Real/") in want_rv).items()})

        def fetch(e):
            z = dirs.get(e["source_id"])
            for k in range(4):
                try:
                    return z[0].read(z[1]) if z else None
                except Exception:
                    time.sleep(3 * (k + 1))
            return None
    print("MEASURE", a.source, len(ents), "todo", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.streams) as net, ProcessPoolExecutor(a.workers) as cpu, dest.open("a") as f:
        for i in range(0, len(ents), 200):
            grp = ents[i:i + 200]
            raws = list(net.map(fetch, grp))
            ok = [(e, r) for e, r in zip(grp, raws) if r is not None
                  and hashlib.sha256(r).hexdigest() == e["locator"]["source_sha256"]]
            res = list(cpu.map(_trim, [r for _, r in ok], chunksize=4))
            for (e, _), (tr, full, err) in zip(ok, res):
                f.write(json.dumps({"source_id": e["source_id"], "verified": True, "trimmed_s": tr,
                                    "decoded_s": full, "error": err, "cache_eligible": e["eligible"]}) + "\n")
            f.flush()
            print(f"MEASURED {i + len(grp)}/{len(ents)} verified={len(ok)}/{len(grp)} {time.time() - t0:.0f}s", flush=True)


def report(a):
    root = Path(a.out)
    trim = {}
    for p in (root / "measure").glob("trim-*.jsonl"):
        src = p.stem.removeprefix("trim-")
        for l in p.read_text().splitlines():
            r = json.loads(l)
            trim[src, r["source_id"]] = r
    pairs = []
    for src, sub, mk in (("rvcbench", "rvcbench", "model"), ("speechfake", "speechfake", "system")):
        f = root / sub / "pairs.jsonl"
        if f.exists():
            for l in f.read_text().splitlines():
                q = json.loads(l)
                pairs.append((src, q[mk], q))
    # Reserved/test text set (VCTK sentence text shared across speakers) for overlap removal.
    reserved_text = {q["text"].strip().lower() for _, _, q in pairs if q.get("text")
                     and q["use"] not in ("train_candidate",)}
    agree = disagree = 0
    rows = defaultdict(lambda: {t: [0, set(), set()] for t in THRESHOLDS})
    removed = defaultdict(int)
    seen_parent_utt = {}
    for src, model, q in sorted(pairs, key=lambda t: (t[0], t[2]["fake"])):
        fk, gk = trim.get((src, q["fake"])), trim.get((src, q["genuine"]))
        if not (fk and gk and fk["trimmed_s"] is not None and gk["trimmed_s"] is not None):
            removed["unverified_or_unreadable"] += 1
            continue
        for k in (fk, gk):
            agree += (k["trimmed_s"] >= 4.0) == k["cache_eligible"]
            disagree += (k["trimmed_s"] >= 4.0) != k["cache_eligible"]
        group = "train" if q["use"] == "train_candidate" else "diagnostic:" + q["use"]
        if group == "train":
            utt = Path(q["genuine"]).stem.replace("_mic1", "")
            if q.get("text") and q["text"].strip().lower() in reserved_text:
                removed["text_overlaps_reserved_or_eval"] += 1
                group = "diagnostic:text_overlap"
            elif seen_parent_utt.setdefault((model, utt), src) != src:
                removed["duplicate_parent_utterance_across_sources"] += 1
                continue
        spk = q.get("parent_speaker", q["speaker"])
        m = min(fk["trimmed_s"], gk["trimmed_s"])
        for t in THRESHOLDS:
            if m >= t:
                c = rows[f"{group}|{src}|{model}"][t]
                c[0] += 1
                c[1].add(spk)
                c[2].add(q["genuine"])
    table = {k: {f">={t:g}s": {"pairs": v[t][0], "speakers": len(v[t][1]), "genuine_parents": len(v[t][2])}
                 for t in THRESHOLDS} for k, v in sorted(rows.items())}
    tot = {}
    for t in THRESHOLDS:
        sp, par, n = set(), set(), 0
        for k, v in rows.items():
            if k.startswith("train|"):
                n += v[t][0]; sp |= v[t][1]; par |= v[t][2]
        tot[f">={t:g}s"] = {"pairs": n, "speakers": len(sp), "genuine_parents": len(par)}
    dist = defaultdict(list)
    for (src, sid), r in trim.items():
        if r["trimmed_s"] is not None:
            dist[src].append(r["trimmed_s"])
    import numpy as np
    rep = {"kind": "measure-only trimmed duration (audio.prepare) and pair eligibility; no cache written",
           "eligible_rule": "pair counts only if both fake and genuine trimmed duration >= threshold",
           "train_total": tot, "by_group_source_model": table, "removed": dict(removed),
           "check_4s_vs_cache_eligibility": {"agree": agree, "disagree": disagree},
           "trimmed_quantiles_s": {s: dict(zip(["p10", "p25", "p50", "p75", "p90"],
                                               np.round(np.percentile(v, [10, 25, 50, 75, 90]), 2).tolist()))
                                   for s, v in dist.items()},
           "objects_measured": {s: len(v) for s, v in dist.items()}}
    (root / "measure" / "trim_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["train_total"]), json.dumps(rep["removed"]), rep["check_4s_vs_cache_eligibility"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["objects", "report"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", choices=["rvcbench", "speechfake"])
    p.add_argument("--streams", type=int, default=6)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    {"objects": objects, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
