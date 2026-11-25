"""V2-B candidate screen: VCC 2018 voice conversion, train-only shortcut audit (diagnostic).

Source: Edinburgh DataShare 10.7488/ds/2337 (CC BY 4.0), files md5-verified against the
publisher REST checksums. Lineage is taken from publisher names and verified per clip:
  converted  testVCC2/<system>_<target>_<source>_<sentence>_<HUB|SPO>.wav
  genuine    vcc2018_reference/<target>/<sentence>.wav   (target speaker, same sentence)
             vcc2018_evaluation/<source>/<sentence>.wav  (source speaker, same sentence)
  text       vcc2018_database_evaluation_transcriptions (sentence id -> text)
A converted clip is admitted only if its genuine target and source recordings and its
transcript all exist. The audit compares converted clips with the GENUINE TARGET recording
of the same sentence (the voice the conversion imitates), using the V1/V2 model input
(4 s fixed crop + RMS norm, clip_cache.treat) and the audit descriptors. Folds leave one
target speaker out (4 folds), so a probe never sees the tested target's genuine speech.
This is a diagnostic, not a gate; outputs go to $D/v2b/vcc2018/.

  python -m pipeline_v3.v2b_vcc --local /content/vcc2018 --data $D --per-system 40
"""

import argparse
import hashlib
import json
import re
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

NAME = re.compile(r"(?P<system>[A-Z]\d+)_(?P<target>VCC2T[FM]\d)_(?P<source>VCC2S[FM]\d)_(?P<sent>\d{5})_(?P<task>HUB|SPO)\.wav$")


def parse(member):
    m = NAME.search(member)
    return m.groupdict() if m else None


def _feat(raw):
    from .audit import descriptors
    from .clip_cache import treat

    b = treat(raw)
    return None if b is None else descriptors(b)


def transcripts(local):
    out = {}
    with tarfile.open(local / "vcc2018_database_evaluation_transcriptions.tar.gz") as t:
        for m in t:
            if m.isfile() and re.fullmatch(r"\d{5}\.txt", Path(m.name).name):
                out[Path(m.name).stem] = t.extractfile(m).read().decode("utf-8", "replace").strip()
    return out


def main():
    from .en_treatment import score
    from .v2a import aucs

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--per-system", type=int, default=40)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    out = a.data / "v2b" / "vcc2018"
    out.mkdir(parents=True, exist_ok=True)
    texts = transcripts(a.local)
    ref = zipfile.ZipFile(a.local / "vcc2018_database_reference.zip")
    ev = zipfile.ZipFile(a.local / "vcc2018_database_evaluation.zip")
    refs = {(Path(n).parent.name, Path(n).stem): n for n in ref.namelist() if n.endswith(".wav")}
    srcs = {(Path(n).parent.name, Path(n).stem): n for n in ev.namelist() if n.endswith(".wav")}
    conv = []
    unparsed = 0
    lineage_fail = Counter()
    with tarfile.open(a.local / "vcc2018_submitted_systems_converted_speech.tar.gz") as t:
        names = [m.name for m in t if m.isfile() and m.name.endswith(".wav")]
    for n in names:
        d = parse(n)
        if not d:
            unparsed += 1
            continue
        ok = {"target_genuine": (d["target"], d["sent"]) in refs,
              "source_genuine": (d["source"], d["sent"]) in srcs, "transcript": d["sent"] in texts}
        for k, v in ok.items():
            lineage_fail[k] += not v
        if all(ok.values()):
            conv.append(dict(d, member=n))
    systems = sorted({c["system"] + "/" + c["task"] for c in conv})
    picked = []
    for s in systems:
        group = sorted((c for c in conv if c["system"] + "/" + c["task"] == s),
                       key=lambda c: hashlib.sha256(("v2b-vcc-v1/" + c["member"]).encode()).hexdigest())
        picked += group[: a.per_system]
    want = {c["member"] for c in picked}
    t0 = time.time()
    raws = {}
    with tarfile.open(a.local / "vcc2018_submitted_systems_converted_speech.tar.gz") as t:
        for m in t:
            if m.name in want:
                raws[m.name] = t.extractfile(m).read()
    genuine = sorted({(c["target"], c["sent"]) for c in conv})
    jobs = [("1", c, raws[c["member"]]) for c in picked] + [("0", {"target": tg, "sent": s}, ref.read(refs[tg, s])) for tg, s in genuine]
    with ProcessPoolExecutor(a.workers) as ex:
        feats = list(ex.map(_feat, [j[2] for j in jobs], chunksize=8))
    rows = [(y, c, f) for (y, c, _), f in zip(jobs, feats) if f is not None]
    targets = sorted({c["target"] for _, c, _ in rows})

    def cv(subset):
        per = []
        for k in targets:
            rr = [{"label": y, "split": "select" if c["target"] == k else "train"} for y, c, _ in subset]
            if {r["label"] for r in rr if r["split"] == "select"} != {"0", "1"}:
                continue
            per.append(aucs(rr, [f for _, _, f in subset]))
        worst, mean = score(per)
        return {"worst": worst, "folds": len(per), "mean_oriented_auc": mean,
                "failed_at_0.60": sorted(k for k, v in mean.items() if v > 0.60)}

    gen = [r for r in rows if r[0] == "0"]
    result = {
        "kind": "V2-B VCC 2018 train-only shortcut diagnostic (leave-one-target-speaker-out; not a gate)",
        "source": "Edinburgh DataShare 10.7488/ds/2337, CC BY 4.0, publisher md5s verified",
        "lineage": {"converted_files": len(names), "unparsed_names": unparsed, "admitted": len(conv),
                    "missing": dict(lineage_fail), "systems": len(systems),
                    "speakers": {"source": len({c["source"] for c in conv}), "target": len({c["target"] for c in conv})},
                    "sentences": len({c["sent"] for c in conv})},
        "sample": {"fakes": sum(r[0] == "1" for r in rows), "genuine_target_refs": len(gen),
                   "dropped_short": len(jobs) - len(rows), "seconds": time.time() - t0},
        "real_call_coverage": "none: DAPS-derived studio-quality read speech; no telephone or VoIP recordings",
        "pooled": cv(rows),
        "by_task": {task: cv(gen + [r for r in rows if r[0] == "1" and r[1]["task"] == task]) for task in ("HUB", "SPO")},
        "by_system": {s: cv(gen + [r for r in rows if r[0] == "1" and r[1]["system"] + "/" + r[1]["task"] == s]) for s in systems},
    }
    (out / "diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("lineage", "sample", "pooled", "by_task")}, indent=2))


if __name__ == "__main__":
    main()
