"""Prepare the English V2-A experiment from existing Drive development archives."""

import argparse
import csv
import hashlib
import json
import os
import tarfile
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from . import data
from .audio import fixed_crop
from .audit import descriptors, probe_scores
from .en_treatment import apply, score
from .match import prompt, short
from .metrics import metrics

FAMILIES = ("melgan", "parallel_wave_gan", "wavegrad", "wavernn")
ASV = "asvspoof5"
LSV = "librisevoc"
VERSION = "v2a-norm-1"
V1 = "stage0-en-ecapa-xtts-norm-exploratory"


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
    os.replace(f.name, path)


def protocol(root):
    with tarfile.open(root / "development/asvspoof5/ASVspoof5_protocols.tar.gz") as arc:
        rows = {}
        for line in arc.extractfile("ASVspoof5.train.tsv"):
            sp, uid, _, _, _, _, _, system, label, _ = line.decode().split()
            if system in {"bonafide", "A08"}:
                if uid in rows or (label, system == "A08") not in {
                    ("spoof", True),
                    ("bonafide", False),
                }:
                    raise ValueError("Unexpected ASVspoof train protocol")
                rows[uid] = (sp, system)
    return rows


def confirmation_sample(rows, screened):
    groups = defaultdict(list)
    for uid, (sp, system) in rows.items():
        if sp not in screened:
            groups[sp, system].append(uid)
    speakers = sorted(
        (
            sp
            for sp, system in groups
            if system == "A08"
            and min(len(groups.get((sp, s), [])) for s in ("A08", "bonafide")) >= 4
        ),
        key=lambda sp: short(VERSION + sp),
    )[:40]
    if len(speakers) != 40:
        raise ValueError("Need 40 fresh confirmation speakers")
    return {
        uid: (sp, system, i % 5)
        for i, sp in enumerate(speakers)
        for system in ("A08", "bonafide")
        for uid in sorted(groups[sp, system], key=short)[:4]
    }


def asv_audio(root, wanted):
    missing = set(wanted)
    for part in "abcde":
        with tarfile.open(
            root / f"development/asvspoof5/flac_T_a{part}.tar", "r:"
        ) as arc:
            for m in arc:
                uid = Path(m.name).stem
                if uid in wanted and m.isfile() and m.name.endswith(".flac"):
                    if uid not in missing:
                        raise ValueError(f"Duplicate audio {uid}")
                    missing.remove(uid)
                    yield uid, arc.extractfile(m).read()
    if missing:
        raise ValueError(f"Missing {len(missing)} requested ASV files")


def treated(b):
    crop = fixed_crop(b)
    return apply(crop, "norm") if crop is not None else None


def aucs(rows, features):
    labels = [int(r["label"]) for r in rows if r["split"] == "select"]
    for split in ("train", "select"):
        if {r["label"] for r in rows if r["split"] == split} != {"0", "1"}:
            raise ValueError(f"Missing audit label in {split}")
    return {
        k: metrics(labels, s)["auc"] for k, s in probe_scores(rows, features).items()
    }


def confirm(root, work):
    old = json.loads((root / "v2/asvspoof5/speaker-cv-v1/sample.json").read_text())
    screened = {v[0] for v in old.values()}
    if len(screened) != 40:
        raise ValueError("Original ASV screen must identify its 40 speakers")
    sample = confirmation_sample(protocol(root), screened)
    # Preserve equal speaker/label counts if a recording is too short.
    cells = defaultdict(list)
    for uid, raw in asv_audio(root, sample):
        x = treated(raw)
        if x is not None:
            sp, system, fold = sample[uid]
            cells[sp, str(int(system == "A08"))].append((uid, fold, descriptors(x)))
    items = []
    for sp in sorted({s for s, _, _ in sample.values()}):
        n = min(len(cells[sp, label]) for label in "01")
        if not n:
            raise ValueError(f"No eligible controls for {sp}")
        for label in "01":
            items.extend(
                (label, fold, feat) for _, fold, feat in sorted(cells[sp, label])[:n]
            )
    results = [
        aucs(
            [
                {"label": y, "split": "select" if fold == k else "train"}
                for y, fold, _ in items
            ],
            [f for _, _, f in items],
        )
        for k in range(5)
    ]
    worst, mean = score(results)
    result = {
        "version": VERSION,
        "sample": sample,
        "eligible": len(items),
        "folds": results,
        "mean_oriented_auc": mean,
        "worst": worst,
        "passed": worst <= 0.60,
    }
    save(work / "confirmation.json", result)
    return result


def requests(root, work):
    """Write join keys for publisher transcripts and the canonical speaker crosswalk."""
    if (work / "lineage-requests.csv").exists():
        return
    records = []
    for r in read_csv(root / "v2/librisevoc/manifest.csv"):
        if r["label"] == "0":
            records.append(
                {
                    "source": LSV,
                    "uid": r["parent"].split("/")[-1],
                    "publisher_speaker": r["speaker"],
                    "speaker": r["speaker"],
                    "transcript": "",
                    "evidence": "",
                }
            )
    for uid, (speaker, _) in sorted(protocol(root).items()):
        records.append(
            {
                "source": ASV,
                "uid": uid,
                "publisher_speaker": speaker,
                "speaker": "",
                "transcript": "",
                "evidence": "",
            }
        )
    with tempfile.NamedTemporaryFile(dir=work, delete=False) as f:
        name = Path(f.name)
    write_catalog(records, name)
    os.replace(name, work / "lineage-requests.csv")


def lineage(path):
    rows = {}
    for r in read_csv(path):
        key = r["source"], r["uid"]
        if key in rows or not all(
            r[k].strip() for k in ("speaker", "transcript", "evidence")
        ):
            raise ValueError(f"Duplicate or unresolved lineage {key}")
        if r["speaker"] == "-" or "|" in r["speaker"]:
            raise ValueError("Expected one verified canonical speaker per input")
        rows[key] = r
    return rows


def candidates(root, work):
    selection = read_csv(root / "stage0-en-ecapa-xtts/selection.csv")
    owners = {r["speaker"]: r["split"] for r in selection if r["label"] == "0"}
    texts = defaultdict(set)
    for r in selection:
        texts[r["text"]].add(r["split"])
    evidence = lineage(work / "lineage.csv")
    confirmation = json.loads((work / "confirmation.json").read_text())
    old = json.loads((root / "v2/asvspoof5/speaker-cv-v1/sample.json").read_text())
    used = {v[0] for v in [*old.values(), *confirmation["sample"].values()]}
    groups = defaultdict(list)
    for r in read_csv(root / "v2/librisevoc/manifest.csv"):
        uid = r["parent"].split("/")[-1]
        if (LSV, uid) in evidence and r["generator"] in ("genuine", *FAMILIES):
            e = evidence[LSV, uid]
            if e["speaker"] != r["speaker"]:
                raise ValueError("LibriSeVoc canonical speaker mismatch")
            groups[LSV, uid].append((r["member"], r["generator"], e))
    asv = protocol(root)
    for uid, (sp, system) in asv.items():
        if (ASV, uid) in evidence:
            e = evidence[ASV, uid]
            groups[ASV, uid].append((uid, system, e))

    # A single publisher speaker must map to exactly one canonical identity.
    asv_people = defaultdict(set)
    for src, uid in groups:
        if src == ASV:
            asv_people[asv[uid][0]].add(evidence[src, uid]["speaker"])
    if any(len(ids) != 1 for ids in asv_people.values()):
        raise ValueError("Contradictory ASV speaker crosswalk")
    screened_people = {person for sp in used for person in asv_people.get(sp, ())}
    planned = []
    for (src, uid), members in sorted(groups.items()):
        if src == LSV and {g for _, g, _ in members} != {"genuine", *FAMILIES}:
            raise ValueError(f"Incomplete four-vocoder parent {uid}")
        for member, system, e in members:
            sp = e["speaker"]
            split = owners.get(sp)
            if split is None and src == ASV:
                split = "select" if int(short(VERSION + sp), 16) % 5 == 0 else "train"
            if split not in {"train", "select"}:
                continue
            if split == "select" and (
                sp in screened_people or (src == ASV and asv[uid][0] in used)
            ):
                continue
            key = prompt(e["transcript"])
            if texts[key] - {split}:
                continue
            fake = system not in {"genuine", "bonafide"}
            rid = short(f"{src}/{member}")
            planned.append(
                {
                    "id": rid,
                    "member": member,
                    "source": src,
                    "pair": src,
                    "speaker": sp,
                    "split": split,
                    "label": str(int(fake)),
                    "voice": sp if fake else "-",
                    "parent": f"{src}/{uid}",
                    "text": key,
                    "language": "en",
                    "channel": "clean",
                    "generator": system if fake else "-",
                    "domain": "seen" if fake else "genuine",
                    "prepared_view": "v2a-fixed-norm",
                    "path": f"audio/{rid}.wav",
                }
            )
    by_text = defaultdict(set)
    for r in planned:
        by_text[r["text"]].add(r["split"])
    return [r for r in planned if len(by_text[r["text"]]) == 1]


def write_catalog(rows, path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def source_panel(rows, features):
    result = {}
    # For the mixture audit use one deterministic fake per LibriSeVoc parent.
    # This reproduces balanced source/label sampling without quadrupling genuine rows.
    by_parent = defaultdict(list)
    for r in rows:
        if r["source"] == LSV and r["label"] == "1":
            by_parent[r["parent"]].append(r)
    rotated = {
        min(group, key=lambda r: short(VERSION + r["id"]))["id"]
        for group in by_parent.values()
    }
    for name in ("pooled", LSV, ASV, *FAMILIES):
        panel = [
            r
            for r in rows
            if name == "pooled"
            or r["source"] == name
            or (
                name in FAMILIES
                and r["source"] == LSV
                and r["generator"] in {name, "-"}
            )
        ]
        if name in {"pooled", LSV}:
            panel = [
                r
                for r in panel
                if r["source"] != LSV or r["label"] == "0" or r["id"] in rotated
            ]
        scores = aucs(panel, [features[r["id"]] for r in panel])
        worst, values = score([scores])
        result[name] = {
            "worst": worst,
            "oriented_auc": values,
            "counts": dict(Counter(r["split"] + "/" + r["label"] for r in panel)),
        }
    return result


def balance(rows):
    cells = defaultdict(list)
    for r in rows:
        unit = r["parent"] if r["source"] == LSV else r["speaker"]
        cells[r["source"], r["split"], unit].append(r)
    kept = []
    for (src, _, _), group in cells.items():
        if src == LSV:
            if Counter(r["generator"] for r in group) == Counter(("-", *FAMILIES)):
                kept.extend(group)
        else:
            labels = [
                sorted((r for r in group if r["label"] == y), key=lambda r: r["id"])
                for y in "01"
            ]
            n = min(map(len, labels))
            for subset in labels:
                kept.extend(subset[:n])
    return kept


def prepare(root, work, scratch):
    confirmation = json.loads((work / "confirmation.json").read_text())
    if confirmation["version"] != VERSION or not confirmation["passed"]:
        raise ValueError("Fresh-speaker A08 confirmation did not pass")
    rows = candidates(root, work)
    preview = balance(rows)
    data.check(preview)
    for src in (LSV, ASV):
        for split in ("train", "select"):
            if not any(r["source"] == src and r["split"] == split for r in preview):
                raise ValueError(
                    f"Lineage does not provide a usable {src}/{split} panel"
                )
    recipe = {
        "version": VERSION,
        "treatment": "norm",
        "families": [*FAMILIES, "A08"],
        "selection_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest(),
        "lineage_sha256": data.digest(work / "lineage.csv"),
        "confirmation_sha256": data.digest(work / "confirmation.json"),
        "threshold": 0.60,
    }
    frozen = work / "recipe.json"
    if frozen.exists() and json.loads(frozen.read_text()) != recipe:
        raise ValueError("Frozen V2-A selection changed")
    save(frozen, recipe)
    features = {}
    dev = scratch / "development"
    (dev / "audio").mkdir(parents=True)

    def store(r, raw):
        b = treated(raw)
        if b is not None:
            (dev / r["path"]).write_bytes(b)
            r["sha256"] = hashlib.sha256(b).hexdigest()
            features[r["id"]] = descriptors(b)

    with zipfile.ZipFile(root / "development/librisevoc/LibriSeVoc.zip") as z:
        for r in rows:
            if r["source"] == LSV:
                store(r, z.read(r["member"]))
    wanted = {r["member"]: r for r in rows if r["source"] == ASV}
    for uid, raw in asv_audio(root, wanted):
        store(wanted[uid], raw)
    rows = balance([r for r in rows if r["id"] in features])
    data.check(rows)
    for split in ("train", "select"):
        for src in (ASV, LSV):
            if len(
                {
                    r["speaker"]
                    for r in rows
                    if r["source"] == src and r["split"] == split
                }
            ) < (20 if split == "train" else 5):
                raise ValueError(f"Insufficient {src} {split} speaker coverage")
    panels = source_panel(rows, features)
    passed = all(v["worst"] <= 0.60 for v in panels.values())
    report = dict(
        version=VERSION,
        policy="v2a-speaker-and-parent-controls-v1",
        languages_audited=["en"],
        panels=panels,
        lineage_sha256=data.digest(work / "lineage.csv"),
        confirmation_sha256=data.digest(work / "confirmation.json"),
        matching={LSV: "exact utterance", ASV: "same speaker, balanced within speaker"},
        **{"pass": passed},
    )
    report["recipe_sha256"] = data.digest(frozen)
    report["audited_rows_sha256"] = hashlib.sha256(
        json.dumps(sorted((r["id"], r["split"], r["sha256"]) for r in rows)).encode()
    ).hexdigest()
    previous = work / "mixture-audit.json"
    if previous.exists() and json.loads(previous.read_text()) != report:
        raise ValueError("Frozen audit inputs or results changed")
    save(previous, report)
    if not passed:
        raise ValueError("Frozen V2-A mixture failed. See mixture-audit.json")
    # Preserve identical V1 comparison audio, including its established held-out folds.
    # V2 select above is exclusively a formal acoustic test, never used to fit the teacher.
    rows = [r for r in rows if r["split"] == "train"]
    old = root / V1 / "packed"
    sums = json.loads((old / "checksums.json").read_text())
    if data.digest(old / "index.json") != sums["index.json"]:
        raise ValueError("V1 reference index checksum mismatch")
    transfer = data.entries(old, data.SPLITS - {"train"})
    select_parents = {
        r["parent"]
        for r in transfer
        if r["split"] == "select" and r["generator"] == "chatterboxtts"
    }
    if not select_parents:
        raise ValueError("Missing unseen Chatterbox selection controls")
    for r in transfer:
        if r["split"] == "select" and (
            r["parent"] not in select_parents
            or r["label"] == "1"
            and r["generator"] != "chatterboxtts"
        ):
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
    counts = Counter(r["label"] for r in rows if r["split"] == "train")
    if min(counts.get(y, 0) for y in "01") < 1000:
        raise ValueError("Insufficient training coverage for pilot")
    write_catalog(rows, dev / "catalog.csv")
    # Recover a disconnect between pack publication and audit.json publication.
    target = work / "packed"
    if target.exists():
        index = json.loads((target / "index.json").read_text())
        expected = {r["id"]: {k: r[k] for k in data.FIELDS + ["sha256"]} for r in rows}
        actual = {r["id"]: {k: r[k] for k in data.FIELDS + ["sha256"]} for r in index}
        if actual != expected or len(index) != len(rows):
            raise ValueError("Partial publication has different content")
        for r in index:
            data.payload(target, r)
    else:
        data.pack(scratch, "catalog.csv", target)
    report["index_sha256"] = data.digest(work / "packed/index.json")
    save(work / "audit.json", report)


def verify_published(work, report, index):
    mixture = json.loads((work / "mixture-audit.json").read_text())
    expected_panels = {"pooled", LSV, ASV, *FAMILIES}
    if (
        report.get("version") != VERSION
        or report.get("policy") != "v2a-speaker-and-parent-controls-v1"
        or report.get("pass") is not True
        or report.get("languages_audited") != ["en"]
        or set(report["panels"]) != expected_panels
        or not all(0.5 <= p["worst"] <= 0.60 for p in report["panels"].values())
        or report != {**mixture, "index_sha256": data.digest(index)}
        or report["recipe_sha256"] != data.digest(work / "recipe.json")
        or report["lineage_sha256"] != data.digest(work / "lineage.csv")
        or report["confirmation_sha256"] != data.digest(work / "confirmation.json")
    ):
        raise ValueError("V2-A audit does not authorize this pack")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["requests", "confirm", "prepare"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--scratch", type=Path, default=Path("/content"))
    a = p.parse_args()
    work = a.data / "v2" / VERSION
    work.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (work / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.action == "requests":
            requests(a.data, work)
        elif a.action == "confirm":
            path = work / "confirmation.json"
            result = (
                json.loads(path.read_text()) if path.exists() else confirm(a.data, work)
            )
            print("A08 confirmation", result["worst"], flush=True)
            if not result["passed"]:
                raise SystemExit("A08 confirmation failed")
        else:
            if (work / "audit.json").exists():
                from .cli import bound_audit

                bound_audit(work, work / "packed/index.json")
                return
            if (work / "mixture-audit.json").exists() and not json.loads(
                (work / "mixture-audit.json").read_text()
            )["pass"]:
                raise ValueError("Frozen mixture already failed; retain formal result")
            with tempfile.TemporaryDirectory(dir=a.scratch) as tmp:
                prepare(a.data, work, Path(tmp))


if __name__ == "__main__":
    main()
