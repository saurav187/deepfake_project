"""Bounded AMI genuine-variety sample (CC BY 4.0) for genuine false-alarm measurement.

Two sessions with disjoint participants. For each participant's transcribed segments
(>= --min-seconds), the same time span is cut from the participant's own headset channel
and from the distant microphone Array1-01. Both siblings share one parent
(session/participant/segment) and therefore one fold; label is genuine (0) for all.
Real meeting rooms, not telephone or RTC transport.

  python -m pipeline_v3.ami_sample download --raw /content/mi/aux/ami
  python -m pipeline_v3.ami_sample cache --raw /content/mi/aux/ami --out /content/mi/v1
"""

import argparse
import hashlib
import io
import json
import re
import time
import urllib.request
import wave
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SESSIONS = ("ES2002a", "IS1009a")
AUDIO = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{s}/audio/{s}.{c}.wav"
ANNOT = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
SOURCE = "ami"
DISTANT = "Array1-01"


def fetch(url, dest, tries=8):
    if dest.exists():
        return
    for k in range(tries):
        try:
            tmp = dest.with_name(dest.name + ".tmp")
            with urllib.request.urlopen(url, timeout=300) as r, tmp.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            return
        except Exception as e:
            print("retry", url, type(e).__name__, flush=True)
            time.sleep(20 * (k + 1))  # mirror returns 503 under rapid requests
    raise RuntimeError("failed " + url)


def download(a):
    a.raw.mkdir(parents=True, exist_ok=True)
    fetch(ANNOT, a.raw / "ami_public_manual_1.6.2.zip")
    for s in SESSIONS:
        for c in [f"Headset-{i}" for i in range(4)] + [DISTANT]:
            fetch(AUDIO.format(s=s, c=c), a.raw / f"{s}.{c}.wav")
            print("got", s, c, flush=True)
            time.sleep(5)
    sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.raw.iterdir()) if p.suffix in (".wav", ".zip")}
    (a.raw / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1))
    print("DOWNLOAD done", len(sums), flush=True)


def speakers(z):
    """meetings.xml -> {session: {agent: (channel, global_name)}}"""
    txt = z.read(next(n for n in z.namelist() if n.endswith("corpusResources/meetings.xml"))).decode()
    out = defaultdict(dict)
    for m in re.finditer(r'<meeting[^>]*observation="([^"]+)"[^>]*>(.*?)</meeting>', txt, re.S):
        for sp in re.finditer(r"<speaker\b([^>]*)/?>", m.group(2)):
            attrs = dict(re.findall(r'([\w:]+)="([^"]*)"', sp.group(1)))
            out[m.group(1)][attrs.get("nxt_agent")] = (attrs.get("channel"), attrs.get("global_name"))
    return out


def segments(z, session, agent):
    name = f"segments/{session}.{agent}.segments.xml"
    names = [n for n in z.namelist() if n.endswith(name)]
    if not names:
        return []
    txt = z.read(names[0]).decode()
    return [(m.group(1), float(m.group(2)), float(m.group(3))) for m in re.finditer(
        r'<segment[^>]*nite:id="([^"]+)"[^>]*transcriber_start="([\d.]+)"[^>]*transcriber_end="([\d.]+)"', txt)]


def words(z, session, agent):
    names = [n for n in z.namelist() if n.endswith(f"words/{session}.{agent}.words.xml")]
    if not names:
        return []
    txt = z.read(names[0]).decode()
    return [(float(m.group(1)), float(m.group(2)), m.group(3)) for m in re.finditer(
        r'<w[^>]*starttime="([\d.]+)"[^>]*endtime="([\d.]+)"[^>]*>([^<]*)</w>', txt)]


def cut(path, t0, t1):
    with wave.open(str(path)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        w.setpos(int(t0 * sr))
        frames = w.readframes(int((t1 - t0) * sr))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as o:
        o.setnchannels(ch)
        o.setsampwidth(sw)
        o.setframerate(sr)
        o.writeframes(frames)
    return buf.getvalue()


def cache(a):
    from .modern_intake import IntakeCache, _treat

    lock = a.out / "writer.lock"
    if lock.exists() and Path("/proc/" + lock.read_text().strip()).exists():
        raise SystemExit("Another cache writer holds the lock")
    import os
    lock.write_text(str(os.getpid()))
    z = zipfile.ZipFile(a.raw / "ami_public_manual_1.6.2.zip")
    spk = speakers(z)
    sums = json.loads((a.raw / "SHA256SUMS.json").read_text())
    jobs, panel = [], []
    for s in SESSIONS:
        for agent, (channel, person) in sorted(spk[s].items()):
            segs = [x for x in segments(z, s, agent) if x[2] - x[1] >= a.min_seconds]
            segs = sorted(segs, key=lambda x: hashlib.sha256(x[0].encode()).hexdigest())[: a.per_speaker]
            ww = words(z, s, agent)
            for seg_id, t0, t1 in sorted(segs, key=lambda x: x[1]):
                text = " ".join(w for b, e, w in ww if b >= t0 and e <= t1).strip()
                parent = f"{s}/{person}/{seg_id}"
                for mic, f in (("headset", f"{s}.Headset-{channel}.wav"), ("distant", f"{s}.{DISTANT}.wav")):
                    sid = f"{parent}/{mic}"
                    jobs.append((sid, f, t0, t1))
                    panel.append(dict(source_id=sid, parent=parent, session=s, participant=person, agent=agent,
                                      mic=mic, file=f, file_sha256=sums.get(f), start=t0, end=t1, text=text,
                                      label="0", use="genuine_false_alarm_panel",
                                      licence="CC BY 4.0 (AMI Meeting Corpus)"))
    c = IntakeCache(a.out)
    todo = [j for j in jobs if (SOURCE, j[0]) not in c]
    with ProcessPoolExecutor(a.workers) as ex:
        for i in range(0, len(todo), 200):
            grp = todo[i:i + 200]
            raws = [cut(a.raw / f, t0, t1) for _, f, t0, t1 in grp]
            res = list(ex.map(_treat, raws, chunksize=4))
            recs = [((SOURCE, sid, {"file": f, "start": t0, "end": t1, "segment_sha256": hashlib.sha256(r).hexdigest(),
                                     "source_duration_s": d}), b, reason)
                    for (sid, f, t0, t1), r, (b, reason, d) in zip(grp, raws, res)]
            print("SHARD", c._publish(recs, "/content/intake-scratch"), len(recs), flush=True)
    for p in panel:
        e = c.entries[(SOURCE, p["source_id"])]
        p.update(eligible=e["eligible"], reason=e["reason"])
    out = a.out / "ami"
    out.mkdir(parents=True, exist_ok=True)
    (out / "panel.jsonl").write_text("".join(json.dumps(p) + "\n" for p in panel))
    cov = Counter((p["session"], p["mic"], p["eligible"]) for p in panel)
    parents_both = {p["parent"] for p in panel if p["eligible"]}
    parents_both = {q for q in parents_both if all(p["eligible"] for p in panel if p["parent"] == q)}
    rep = {"sessions": SESSIONS, "participants": {s: sorted(v[1] for v in spk[s].values()) for s in SESSIONS},
           "participants_disjoint": not (set(v[1] for v in spk[SESSIONS[0]].values()) & set(v[1] for v in spk[SESSIONS[1]].values())),
           "coverage": {f"{s}|{m}|{'eligible' if e else 'excluded'}": n for (s, m, e), n in sorted(cov.items())},
           "parents_with_both_mics_eligible": len(parents_both),
           "channels": "own headset vs distant Array1-01 (same span); room microphones, not telephone/RTC",
           "files_sha256": sums}
    (out / "lineage.json").write_text(json.dumps(rep, indent=1))
    print("AMI", json.dumps(rep["coverage"]), "both", len(parents_both), flush=True)
    lock.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["download", "cache"])
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--min-seconds", type=float, default=5.0)
    p.add_argument("--per-speaker", type=int, default=25)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    {"download": download, "cache": cache}[a.cmd](a)


if __name__ == "__main__":
    main()
