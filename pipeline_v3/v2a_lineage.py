"""Resolve V2-A lineage by script: publisher transcripts and canonical speakers.

LibriSeVoc: utterance keys are LibriTTS ids (reader_chapter_paragraph_sentence). The
transcript is LibriTTS train-clean-100 `<uid>.normalized.txt` (OpenSLR 60, md5-verified
archive). LibriTTS keeps LibriSpeech reader ids, and the archive path
train-clean-100/<reader>/<chapter>/<uid>.* is checked against the key, so the canonical
speaker is librispeech/<reader>.

ASVspoof 5: the public release (README + protocols) has no transcripts and only anonymised
T_**** speakers with no MLS reader mapping. No ASV row is admitted unless a verified external
table is supplied with --asv-table (columns uid,speaker,transcript,evidence).

  python -m pipeline_v3.v2a_lineage --data $D --libritts-text /content/libritts-text \
      --archive-md5 4a8c202b78fe1bc0c47916a98f3a2ea8
"""

import argparse
import csv
import fcntl
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

LIBRITTS = "openslr60/train-clean-100.tar.gz"
FIELDS = ("source", "uid", "speaker", "transcript", "evidence")


def resolve(requests, text_root, md5, asv_table=None):
    texts = {}
    for p in Path(text_root).rglob("*.normalized.txt"):
        uid = p.name[: -len(".normalized.txt")]
        reader, chapter = uid.split("_")[:2]
        if p.parent.name != chapter or p.parent.parent.name != reader:
            raise ValueError(f"LibriTTS path does not match key {p}")
        if uid in texts:
            raise ValueError(f"Duplicate LibriTTS transcript {uid}")
        texts[uid] = (p.read_text(encoding="utf-8").strip(), p.relative_to(text_root))
    asv = {}
    if asv_table:
        with open(asv_table) as f:
            for r in csv.DictReader(f):
                if r["uid"] in asv or not all(r[k].strip() for k in ("speaker", "transcript", "evidence")):
                    raise ValueError(f"Bad ASV lineage row {r['uid']}")
                asv[r["uid"]] = r
    out, missing = [], Counter()
    for r in requests:
        if r["source"] == "librisevoc":
            hit = texts.get(r["uid"])
            reader = r["uid"].split("_")[0]
            if not hit or not hit[0] or r["speaker"] != "librispeech/" + reader:
                missing["librisevoc"] += 1
                continue
            out.append({"source": "librisevoc", "uid": r["uid"], "speaker": r["speaker"],
                        "transcript": hit[0],
                        "evidence": f"{LIBRITTS} md5 {md5}: {hit[1]}; speaker = LibriTTS/LibriSpeech reader {reader}"})
        elif r["source"] == "asvspoof5":
            e = asv.get(r["uid"])
            if not e:
                missing["asvspoof5"] += 1
                continue
            out.append({"source": "asvspoof5", "uid": r["uid"], "speaker": e["speaker"],
                        "transcript": e["transcript"], "evidence": e["evidence"]})
        else:
            raise ValueError("Unknown source " + r["source"])
    return out, missing


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--libritts-text", type=Path, required=True)
    p.add_argument("--archive-md5", required=True)
    p.add_argument("--asv-table", type=Path)
    p.add_argument("--work", default="v2a-norm-1", help="experiment dir under $D/v2 (v2l-lsv-norm-1 for V2-L)")
    p.add_argument("--requests", type=Path, help="lineage-requests.csv (default: in --work)")
    a = p.parse_args()
    work = a.data / "v2" / a.work
    work.mkdir(parents=True, exist_ok=True)
    with (a.requests or work / "lineage-requests.csv").open() as f:
        requests = list(csv.DictReader(f))
    if a.work.startswith("v2l"):
        requests = [r for r in requests if r["source"] == "librisevoc"]
    out, missing = resolve(requests, a.libritts_text, a.archive_md5, a.asv_table)
    requested = Counter(r["source"] for r in requests)
    admitted = Counter(r["source"] for r in out)
    report = {
        "requested": dict(requested),
        "admitted": dict(admitted),
        "unresolved": dict(missing),
        "admitted_speakers": {s: len({r["speaker"] for r in out if r["source"] == s}) for s in admitted},
        "asvspoof5_note": "Public ASVspoof 5 release has no transcripts and no MLS reader mapping "
        "(README, ASVspoof5.train.tsv: anonymised T_**** speakers and file names).",
    }
    with (work / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # wait for any running V2-A writer
        if (work / "lineage.csv").exists():
            raise SystemExit("lineage.csv already exists; not overwritten")
        with tempfile.NamedTemporaryFile("w", dir=work, delete=False, newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(out)
        os.replace(f.name, work / "lineage.csv")
        (work / "lineage-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
