"""Bounded AMI genuine-variety sample (CC BY 4.0) for genuine false-alarm measurement.

Source: the Edinburgh CSTR release edinburghcstr/ami on Hugging Face (pinned revision), test
split, segment level. The groups.inf.ed.ac.uk mirror returned HTTP 503 to both the VM and the
workstation on 26 Nov 2026, so this uses the same corpus' official segment export instead.

Two meetings with disjoint participants. A segment is admitted only when the same
(meeting, speaker, begin, end) exists in both the IHM (participant's own headset) and SDM
(distant Array1-01) configs; both siblings share one parent and hence one fold. Label genuine.
Real meeting-room microphones, not telephone or RTC transport.

  python -m pipeline_v3.ami_sample --out /content/mi/v1 --raw /content/mi/aux/ami
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = "edinburghcstr/ami"
REVISION = "46f28f2503e2ec48f8867a84eef356c70476beab"
MEETINGS = ("ES2004a", "IS1009a")
SOURCE = "ami"
META_COLS = ["meeting_id", "audio_id", "text", "begin_time", "end_time", "microphone_id", "speaker_id"]


def shards(fs, config):
    return sorted(p for p in fs.ls(f"datasets/{REPO}@{REVISION}/{config}", detail=False) if "/test-" in p)


def index(fs, config):
    import pyarrow.parquet as pq

    out = {}
    for path in shards(fs, config):
        with fs.open(path, "rb", block_size=1 << 20) as f:
            t = pq.read_table(f, columns=META_COLS).to_pylist()
        for i, r in enumerate(t):
            if r["meeting_id"] in MEETINGS:
                key = (r["meeting_id"], r["speaker_id"], round(r["begin_time"], 2), round(r["end_time"], 2))
                out[key] = dict(r, shard=path.rsplit("/", 1)[1], row=i)
    return out


def audio_bytes(fs, config, shard, rows, raw):
    import pyarrow.parquet as pq

    local = raw / config / shard
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        fs.get(f"datasets/{REPO}@{REVISION}/{config}/{shard}", str(local))
    t = pq.read_table(local, columns=["audio"])
    col = t.column("audio")
    return {i: col[i].as_py()["bytes"] for i in rows}, hashlib.sha256(local.read_bytes()).hexdigest()


def main():
    from huggingface_hub import HfFileSystem

    from .modern_intake import IntakeCache, _treat

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--min-seconds", type=float, default=5.0)
    p.add_argument("--per-speaker", type=int, default=25)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    fs = HfFileSystem()
    ihm, sdm = index(fs, "ihm"), index(fs, "sdm")
    both = sorted(k for k in ihm if k in sdm and k[3] - k[2] >= a.min_seconds)
    spk = defaultdict(set)
    for m, s, _, _ in both:
        spk[m].add(s)
    picked = []
    for m in MEETINGS:
        for s in sorted(spk[m]):
            ks = sorted((k for k in both if k[0] == m and k[1] == s),
                        key=lambda k: hashlib.sha256(repr(k).encode()).hexdigest())[: a.per_speaker]
            picked += ks
    print("INDEX ihm", len(ihm), "sdm", len(sdm), "matched>=min", len(both), "picked", len(picked), flush=True)
    lock = a.out / "writer.lock"
    if lock.exists() and Path("/proc/" + lock.read_text().strip()).exists():
        raise SystemExit("Another cache writer holds the lock")
    lock.write_text(str(os.getpid()))
    c = IntakeCache(a.out)
    jobs, panel, shard_sha = [], [], {}
    for config, idx, mic in (("ihm", ihm, "headset"), ("sdm", sdm, "distant")):
        need = defaultdict(list)
        for k in picked:
            need[idx[k]["shard"]].append(idx[k]["row"])
        for shard, rows in sorted(need.items()):
            got, sha = audio_bytes(fs, config, shard, rows, a.raw)
            shard_sha[f"{config}/{shard}"] = sha
            for k in picked:
                r = idx[k]
                if r["shard"] == shard:
                    parent = f"{k[0]}/{k[1]}/{k[2]:.2f}-{k[3]:.2f}"
                    sid = f"{parent}/{mic}"
                    jobs.append((sid, got[r["row"]], {"repo": REPO, "revision": REVISION, "config": config,
                                                       "shard": shard, "row": r["row"], "audio_id": r["audio_id"],
                                                       "microphone_id": r["microphone_id"]}))
                    panel.append(dict(source_id=sid, parent=parent, meeting=k[0], speaker=k[1], mic=mic,
                                      microphone_id=r["microphone_id"], begin=k[2], end=k[3], text=r["text"],
                                      label="0", use="genuine_false_alarm_panel", upstream_split="test",
                                      licence="CC BY 4.0 (AMI Meeting Corpus)"))
    todo = [j for j in jobs if (SOURCE, j[0]) not in c]
    with ProcessPoolExecutor(a.workers) as ex:
        for i in range(0, len(todo), 200):
            grp = todo[i:i + 200]
            res = list(ex.map(_treat, [g[1] for g in grp], chunksize=4))
            recs = [((SOURCE, sid, dict(loc, source_sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                                          source_duration_s=d)), b, reason)
                    for (sid, raw, loc), (b, reason, d) in zip(grp, res)]
            print("SHARD", c._publish(recs, "/content/intake-scratch"), len(recs), flush=True)
    for q in panel:
        e = c.entries[(SOURCE, q["source_id"])]
        q.update(eligible=e["eligible"], reason=e["reason"])
    out = a.out / "ami"
    out.mkdir(parents=True, exist_ok=True)
    (out / "panel.jsonl").write_text("".join(json.dumps(q) + "\n" for q in panel))
    ok = defaultdict(set)
    for q in panel:
        if q["eligible"]:
            ok[q["parent"]].add(q["mic"])
    rep = {"source": {"repo": REPO, "revision": REVISION, "split": "test", "licence": "CC BY 4.0"},
           "mirror_status": "groups.inf.ed.ac.uk AMICorpusMirror returned HTTP 503 (VM and workstation), 26 Nov 2026",
           "meetings": MEETINGS, "speakers": {m: sorted(spk[m]) for m in MEETINGS},
           "speakers_disjoint": not (spk[MEETINGS[0]] & spk[MEETINGS[1]]),
           "coverage": {f"{m}|{mic}|{'eligible' if e else 'excluded'}": n for (m, mic, e), n in
                        sorted(Counter((q["meeting"], q["mic"], q["eligible"]) for q in panel).items())},
           "parents": len({q["parent"] for q in panel}),
           "parents_both_mics_eligible": sum(len(v) == 2 for v in ok.values()),
           "parquet_sha256": shard_sha,
           "channels": "IHM own headset vs SDM Array1-01 (same span); room microphones, not telephone/RTC"}
    (out / "lineage.json").write_text(json.dumps(rep, indent=1))
    print("AMI", json.dumps(rep["coverage"]), "both", rep["parents_both_mics_eligible"], flush=True)
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
