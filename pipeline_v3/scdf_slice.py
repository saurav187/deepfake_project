"""SCDF English speaker-controlled diagnostic slice (Zenodo 10.5281/zenodo.16744845, CC0).

10 English VoxPopuli speakers (dev/test-derived; F5-TTS was fine-tuned on a train-derived split).
No exact target-speaker + normalized-text join to the genuine subset exists, so this is a
speaker-controlled descriptive diagnostic, never a training source. Per speaker: every genuine
clip with duration >= --min-seconds, and --per-tool hash-ordered fakes per tool with the same
duration floor. Members are range-read from waveforms.zip (48.1 GB, not downloaded whole).
Family notes: XTTS-v2 is a V1 training family; F5-TTS and OpenVoice v2 also occur in RVCBench.

  python -m pipeline_v3.scdf_slice --out /content/mi/v1 --meta /content/mi/aux/scdf_metadata.zip
"""

import argparse
import csv
import hashlib
import io
import json
import os
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from .modern_intake import IntakeCache, _feat, _treat
from .remote_zip import RemoteZip

RECORD = "16744845"
META_MD5 = "58cc9c5b1a95d24d7de8c27981812c33"
WAVES = (f"https://zenodo.org/api/records/{RECORD}/files/waveforms.zip/content", 48111124942)
SOURCE = "scdf"
TOOLS = {"0": "XTTS-v2 (V1 training family)", "1": "OpenVoice v2", "2": "DDDM-VC", "3": "F5-TTS (fine-tuned)"}


def main():
    from .en_treatment import score
    from .v2a import aucs

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--per-tool", type=int, default=20)
    p.add_argument("--min-seconds", type=float, default=4.5)
    p.add_argument("--streams", type=int, default=6)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    log = lambda m: print(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + m, flush=True)
    if hashlib.md5(a.meta.read_bytes()).hexdigest() != META_MD5:
        raise ValueError("SCDF metadata.zip differs from the Zenodo record")
    rows = list(csv.DictReader(io.TextIOWrapper(zipfile.ZipFile(a.meta).open("metadata_en.csv"), encoding="utf-8")))
    by = defaultdict(list)
    for r in rows:
        if float(r["duration"]) >= a.min_seconds:
            by[r["speaker_id"], r["tool"]].append(r)
    picked = []
    for (spk, tool), rr in sorted(by.items()):
        rr = sorted(rr, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        picked += rr if tool == "real" else rr[: a.per_tool]
    lock = a.out / "writer.lock"
    if lock.exists() and Path("/proc/" + lock.read_text().strip()).exists():
        raise SystemExit("Another cache writer holds the lock")
    lock.write_text(str(os.getpid()))
    c = IntakeCache(a.out)
    rz = RemoteZip([WAVES])
    ids = {r["id"] for r in picked}
    t0 = time.time()
    d = rz.directory(lambda n: n.rsplit("/", 1)[-1] in ids and "/en/" in "/" + n)
    names = {n.rsplit("/", 1)[-1]: (n, v) for n, v in d.items()}
    log(f"DIRECTORY {len(names)}/{len(ids)} entries={rz.n_entries} {time.time() - t0:.0f}s")
    todo = [r for r in picked if (SOURCE, r["id"]) not in c]

    def get(r):
        if r["id"] not in names:
            return r, None, "member_not_in_archive_directory"
        try:
            return r, rz.read(names[r["id"]][1]), None
        except Exception as e:
            return r, None, f"download_error:{type(e).__name__}"

    with ThreadPoolExecutor(a.streams) as net, ProcessPoolExecutor(a.workers) as cpu:
        for i in range(0, len(todo), 200):
            got = list(net.map(get, todo[i:i + 200]))
            ok = [g for g in got if g[1] is not None]
            res = list(cpu.map(_treat, [g[1] for g in ok], chunksize=4))
            recs = [((SOURCE, r["id"], {"record": RECORD, "member": names[r["id"]][0], "zip_crc32": names[r["id"]][1][5],
                                         "source_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                                         "source_duration_s": dur}), b, reason)
                    for (r, raw, _), (b, reason, dur) in zip(ok, res)]
            recs += [((SOURCE, r["id"], {"record": RECORD}), None, e) for r, raw, e in got
                     if raw is None and e == "member_not_in_archive_directory"]
            if recs:
                log(f"SHARD {c._publish(recs, '/content/intake-scratch')} +{len(recs)} "
                    f"eligible={sum(x[1] is not None for x in recs)}")
    lock.unlink(missing_ok=True)
    # Speaker-controlled diagnostic: 5 folds of 2 speakers; per tool vs that speaker's genuine.
    elig = [r for r in picked if (SOURCE, r["id"]) in c and c.entries[(SOURCE, r["id"])]["eligible"]]
    with ProcessPoolExecutor(a.workers) as ex:
        feats = dict(zip([r["id"] for r in elig], ex.map(_feat, [c.get(SOURCE, r["id"]) for r in elig], chunksize=8)))
    spk = sorted({r["speaker_id"] for r in elig}, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    fold = {s: k % 5 for k, s in enumerate(spk)}

    def cv(sub):
        per = []
        for k in range(5):
            rr = [{"label": "0" if r["tool"] == "real" else "1", "split": "select" if fold[r["speaker_id"]] == k else "train"} for r in sub]
            if all({x["label"] for x in rr if x["split"] == s} == {"0", "1"} for s in ("train", "select")):
                per.append(aucs(rr, [feats[r["id"]] for r in sub]))
        if not per:
            return {"folds": 0}
        worst, mean = score(per)
        return {"folds": len(per), "worst": worst, "mean_oriented_auc": mean, "n": len(sub),
                "failed_at_0.60": sorted(f for f, v in mean.items() if v > 0.60)}

    real = [r for r in elig if r["tool"] == "real"]
    report = {"kind": "SCDF English speaker-controlled descriptive diagnostic (not a training source)",
              "source": {"zenodo": RECORD, "metadata_md5": META_MD5, "licence": "CC0 1.0"},
              "speakers": len(spk), "genuine_eligible": len(real),
              "picked": dict(Counter(TOOLS.get(r["tool"], "genuine") for r in picked)),
              "eligible": dict(Counter(TOOLS.get(r["tool"], "genuine") for r in elig)),
              "pooled": cv(elig),
              "per_tool": {TOOLS[t]: cv(real + [r for r in elig if r["tool"] == t]) for t in TOOLS},
              "seconds": time.time() - t0}
    out = a.out / "scdf"
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.jsonl").write_text("".join(json.dumps({**r, "use": "diagnostic_speaker_controlled"}) + "\n" for r in picked))
    (out / "diagnostic.json").write_text(json.dumps(report, indent=1))
    log(f"SCDF {json.dumps(report['eligible'])} pooled_worst={report['pooled'].get('worst')}")


if __name__ == "__main__":
    main()
