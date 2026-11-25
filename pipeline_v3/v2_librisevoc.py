"""V2 LibriSeVoc intake: matched manifest, V1 reader overlap, train-only shortcut diagnostic.

V2 data never enters V1 training, calibration or evaluation. All outputs go to
$D/v2/librisevoc/. The source is the publisher ZIP verified on Drive (md5 bee5261d...).

  manifest   genuine gt/<uid>.wav matched to <vocoder>/<uid>_gen.wav (six vocoders) by the
             LibriTTS utterance id; reader = first id field (LibriSpeech reader namespace).
             Reports reader and chapter overlap with every V1 fold and LibriSpeech dev.
  diagnose   a deterministic sample restricted to readers that are V1 TRAIN readers (so V1
             held-out readers stay untouched), extracted to local scratch, the V1 model input
             (audio.fixed_crop 4 s crop, and the frozen `norm` treatment) and the V1 audit
             descriptors; 5-fold reader-grouped CV (parents with their reconstructions),
             probes = audit.fit_controls. Per vocoder vs its own genuine parents, and pooled.

  python -m pipeline_v3.v2_librisevoc manifest --data $D
  python -m pipeline_v3.v2_librisevoc diagnose --data $D --scratch /content/v2scratch
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ZIP = "development/librisevoc/LibriSeVoc.zip"
MD5 = "bee5261df62de16ef900861358afae08"
OUT = "v2/librisevoc"
VOCODERS = ("diffwave", "melgan", "parallel_wave_gan", "wavegrad", "wavenet", "wavernn")
READERS = 40
PER_READER = 8
VIEWS = ("crop", "norm")


def _h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def manifest(data):
    from .stage0 import read, write_csv

    names = zipfile.ZipFile(data / ZIP).namelist()
    gt = {}
    fakes = defaultdict(dict)
    for n in names:
        parts = n.split("/")
        if len(parts) != 3 or not parts[2].endswith(".wav"):
            continue
        _, system, f = parts
        if system == "gt":
            gt[f[:-4]] = n
        elif system in VOCODERS and f.endswith("_gen.wav"):
            fakes[f[: -len("_gen.wav")]][system] = n
    rows = []
    for uid, member in sorted(gt.items()):
        reader, chapter = uid.split("_")[:2]
        if set(fakes.get(uid, {})) != set(VOCODERS):
            continue
        base = dict(parent="librisevoc/" + uid, speaker="librispeech/" + reader, chapter=chapter)
        rows.append(dict(base, id="lsv/gt/" + uid, label="0", generator="genuine", member=member))
        for v in VOCODERS:
            rows.append(dict(base, id=f"lsv/{v}/{uid}", label="1", generator=v, member=fakes[uid][v]))
    unmatched = {"gt_without_all_six": len(gt) - len(rows) // 7,
                 "fake_ids_without_gt": len(set(fakes) - set(gt))}
    # V1 folds: the exploratory selection; LibriSpeech dev panel readers.
    v1 = read(data / "stage0-en-ecapa-xtts" / "selection.csv")
    v1_readers = defaultdict(set)
    v1_chapters = defaultdict(set)
    for r in v1:
        if r["label"] == "0":
            v1_readers[r["split"]].add(r["speaker"])
            reader = r["speaker"].split("/")[-1]
            v1_chapters[r["split"]].add((reader, r["parent"].split("/")[-1].split("-")[1]))
    inv = read(data / "stage0" / "inventory.csv")
    dev = {"librispeech/" + r["reader"] for r in inv if r["group"] in {"dev-clean", "dev-other"}}
    pdf_all = {"librispeech/" + r["reader"] for r in inv if r["group"] == "real"}
    lsv_readers = {r["speaker"] for r in rows}
    lsv_chapters = {(r["speaker"].split("/")[-1], r["chapter"]) for r in rows}
    overlap = {
        "librisevoc_genuine_matched": len(rows) // 7,
        "librisevoc_readers": len(lsv_readers),
        "unmatched": unmatched,
        "reader_overlap_by_v1_fold": {s: len(lsv_readers & v) for s, v in sorted(v1_readers.items())},
        "v1_fold_readers": {s: len(v) for s, v in sorted(v1_readers.items())},
        "chapter_overlap_by_v1_fold": {s: len(lsv_chapters & v) for s, v in sorted(v1_chapters.items())},
        "reader_overlap_librispeech_dev_panel": len(lsv_readers & dev),
        "reader_overlap_phonemedf_all_real": len(lsv_readers & pdf_all),
        "readers_not_in_any_v1_source": len(lsv_readers - pdf_all - dev),
        "note": "LibriTTS ids are reader_chapter_paragraph_sentence; LibriSpeech ids are "
        "reader-chapter-utterance, so overlap is checked at reader and chapter level. "
        "Utterance-level text overlap is not established here.",
    }
    out = data / OUT
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "manifest.csv", rows)
    (out / "overlap.json").write_text(json.dumps(overlap, indent=2))
    return overlap


def sample(rows, train_readers, readers=READERS, per_reader=PER_READER):
    by = defaultdict(list)
    for r in rows:
        if r["label"] == "0" and r["speaker"] in train_readers:
            by[r["speaker"]].append(r["parent"])
    chosen = sorted(by, key=lambda s: _h("v2-lsv-v1/" + s))[:readers]
    parents = {p for s in chosen for p in sorted(by[s], key=lambda p: _h("v2-lsv-v1/" + p))[:per_reader]}
    return [r for r in rows if r["parent"] in parents]


_ZIP = {}


def _zip(path):
    """One open ZIP (one central-directory read) per worker process."""
    if path not in _ZIP:
        _ZIP[path] = zipfile.ZipFile(path)
    return _ZIP[path]


def _work(args):
    zpath, scratch, r = args
    from .audio import fixed_crop
    from .audit import PROVENANCE_FIELDS, descriptors
    from .en_treatment import apply

    b = _zip(zpath).read(r["member"])
    local = Path(scratch) / (r["id"].replace("/", "__") + ".wav")
    local.write_bytes(b)
    crop = fixed_crop(b)
    if crop is None:
        return r["id"], None
    out = {}
    for view in VIEWS:
        x = crop if view == "crop" else apply(crop, "norm")
        out[view] = {k: v for k, v in descriptors(x).items() if k not in PROVENANCE_FIELDS}
    return r["id"], out


def diagnose(data, scratch, workers=None):
    import numpy as np

    from .audit import fit_controls
    from .en_treatment import folds, score
    from .stage0 import read, write_csv

    out = data / OUT
    rows = read(out / "manifest.csv")
    v1 = read(data / "stage0-en-ecapa-xtts" / "selection.csv")
    train_readers = {r["speaker"] for r in v1 if r["split"] == "train" and r["label"] == "0"}
    picked = sample(rows, train_readers)
    z = zipfile.ZipFile(data / ZIP)
    need = sum(z.getinfo(r["member"]).file_size for r in picked)
    print(f"sample {len(picked)} clips, {need/1e9:.2f} GB to extract", flush=True)
    probe = scratch
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < need + 20e9:
        raise ValueError(f"Scratch space too small: need {need/1e9:.1f} GB + 20 GB reserve, free {free/1e9:.1f} GB")
    scratch.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2))
    t0 = time.time()
    feats = {}
    with ProcessPoolExecutor(workers) as ex:
        for i, (rid, f) in enumerate(ex.map(_work, [(str(data / ZIP), str(scratch), r) for r in picked], chunksize=8), 1):
            feats[rid] = f
            if i % 100 == 0 or i == len(picked):
                rate = i / (time.time() - t0)
                print(f"PROGRESS {i}/{len(picked)} clips {rate:.2f}/s", flush=True)
    elapsed = time.time() - t0
    kept = [r for r in picked if feats[r["id"]] is not None]
    ok_parents = {r["parent"] for r in kept if r["label"] == "0"}
    kept = [r for r in kept if r["parent"] in ok_parents]
    for r in kept:
        r.update(split="train", pair="librisevoc", language="en", channel="clean")
    fold_of = folds(kept)

    def cv(subset, view):
        f = [feats[r["id"]][view] for r in subset]
        per_fold = []
        for k in range(5):
            split = [dict(r, split="select" if fold_of[r["id"]] == k else "train") for r in subset]
            rep = fit_controls(split, f)
            per_fold.append({x: rep["controls"][x]["auc"] for x in rep["blocking_fields"]})
        worst, mean = score(per_fold)
        return {"worst": worst, "mean_oriented_auc": mean,
                "failed_at_0.60": sorted(k for k, v in mean.items() if v > 0.60)}

    result = {
        "kind": "V2 LibriSeVoc train-only shortcut diagnostic (V1 TRAIN readers only; not a gate)",
        "sample": {"readers": len({r["speaker"] for r in kept}), "parents": len(ok_parents),
                   "clips": len(kept), "requested_clips": len(picked),
                   "too_short_for_4s_crop": len(picked) - sum(feats[r["id"]] is not None for r in picked),
                   "bytes_read": need},
        "throughput": {"clips_per_s": len(picked) / elapsed, "seconds": elapsed, "workers": workers},
        "views": {},
    }
    for view in VIEWS:
        result["views"][view] = {"pooled_all_vocoders": cv(kept, view)}
        for v in VOCODERS:
            sub = [r for r in kept if r["generator"] in ("genuine", v)]
            result["views"][view][v] = cv(sub, view)
    means = {}
    for view in VIEWS:
        means[view] = {
            g: {k: float(np.mean([feats[r["id"]][view][k] for r in kept if r["generator"] == g]))
                for k in ("rms", "upper_band", "rolloff_hz", "tilt", "speech_ratio")}
            for g in ("genuine", *VOCODERS)
        }
    result["descriptor_means"] = means
    write_csv(out / "diag-sample.csv", kept)
    (out / "diagnostic.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["manifest", "diagnose"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--scratch", type=Path, default=Path("/content/v2scratch/librisevoc"))
    p.add_argument("--workers", type=int)
    a = p.parse_args()
    r = manifest(a.data) if a.action == "manifest" else diagnose(a.data, a.scratch, a.workers)
    print(json.dumps(r, indent=2)[:6000])


if __name__ == "__main__":
    main()
