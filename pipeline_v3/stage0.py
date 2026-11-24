"""PhonemeDF preparation stages. All writes are below DATA, independent of sealed pools."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .adapters import selected_members
from .audio import fixed_crop
from .audit import POLICY, descriptors, panels
from .data import FIELDS, check, digest, pack, safe_path
from .phonemedf import SYSTEMS, counts, inventory, match, proof, write_csv


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def groups(root, rows):
    result = defaultdict(list)
    for r in rows:
        result[safe_path(root, r["archive"])].append(r)
    return result


def transformed(root, rows):
    for r, b in selected_members(groups(root, rows)):
        yield r, fixed_crop(b) if r["pair"] == "phonemedf" else b


def viable(rows):
    check(rows)
    for fold in ("train", "select", "calibrate", "threshold", "validate"):
        en = [
            r
            for r in rows
            if r["language"] == "en" and r["split"] == fold and r["pair"] == "phonemedf"
        ]
        if len({r["speaker"] for r in en}) < (20 if fold == "train" else 5):
            raise ValueError("Insufficient PhonemeDF readers in " + fold)
        need = {"xtts_v2", "melotts"} if fold == "train" else {"chatterboxtts"}
        if not need <= {r["generator"] for r in en}:
            raise ValueError("Missing English generator cell in " + fold)


def audit_selection(root, work):
    rows = read(work / "selection.csv")
    hashes = {}
    features = {}
    for r, b in transformed(root, rows):
        if b is None:
            continue
        hashes[r["id"]] = hashlib.sha256(b).hexdigest()
        if r["split"] in {"train", "select"}:
            features[r["id"]] = dict(descriptors(b), source=r["source"], pair=r["pair"])
    available = [r for r in rows if r["id"] in hashes]

    def parents(label):
        return {r["parent"] for r in available if r["label"] == label}

    joined = parents("0") & parents("1")
    kept = [dict(r, sha256=hashes[r["id"]]) for r in available if r["parent"] in joined]
    viable(kept)
    idx = [r for r in kept if r["id"] in features]
    report = panels(idx, [features[r["id"]] for r in idx])
    report.update(
        selection_sha256=digest(work / "selection.csv"),
        dropped=len(rows) - len(kept),
        counts=counts(kept),
    )
    write_csv(work / "eligible.csv", kept)
    report["eligible_sha256"] = digest(work / "eligible.csv")
    tmp = work / "audit.tmp"
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(work / "prep-audit.json")
    if not report["pass"]:
        raise ValueError(f"Nuisance gate failed; see {work / 'prep-audit.json'}")


def catalog(root, work):
    report = json.loads((work / "prep-audit.json").read_text())
    if (
        report.get("policy") != POLICY
        or not report["pass"]
        or digest(work / "eligible.csv") != report["eligible_sha256"]
        or digest(work / "selection.csv") != report["selection_sha256"]
    ):
        raise ValueError("Passing audit required for this selection")
    rows = read(work / "eligible.csv")
    dest = root / work.name / "bytes"
    dest.mkdir(parents=True, exist_ok=True)
    for r, b in transformed(root, rows):
        if b is None or hashlib.sha256(b).hexdigest() != r["sha256"]:
            raise ValueError("Source changed after audit")
        path = dest / (r["id"] + ".audio")
        if not path.exists() or digest(path) != r["sha256"]:
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(b)
            tmp.replace(path)
        r["path"] = str(path.relative_to(root))
    write_csv(
        work / "catalog.csv", [{k: r[k] for k in [*FIELDS, "sha256"]} for r in rows]
    )


def load_labels(work, rows):
    expected = {(r["group"], r["uid"]) for r in rows if r["group"] in SYSTEMS}
    if (
        digest(work / "inventory.csv")
        != (work / "voice-inventory.sha256").read_text().strip()
    ):
        raise ValueError("Voice inventory mismatch")
    labels = read(work / "voices.csv")
    result = {(r["group"], r["uid"]): r["voice"] for r in labels}
    if set(result) != expected or len(labels) != len(result):
        raise ValueError("Incomplete or duplicated labels")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=["proof", "inventory", "match", "audit", "catalog", "pack", "jobs"],
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--base", default="selection_v0_english_thin.csv")
    p.add_argument("--listings", type=Path)
    p.add_argument("--real-texts", type=Path)
    p.add_argument("--real-glob", default="*.wav")
    p.add_argument("--release", action="store_true")
    p.add_argument("--addons", action="store_true")
    p.add_argument("--microsoft", choices=["azure", "edge"], default="azure")
    a = p.parse_args()
    root = a.data / "development"
    work = a.data / ("stage5" if a.release else "stage3" if a.addons else "stage0")
    if a.action == "proof":
        rows = inventory(None, a.listings, a.real_texts, a.real_glob)
        proof(rows, [r for r in read(Path(a.base)) if r["language"] == "hi"])
        return
    work.mkdir(parents=True, exist_ok=True)
    shared = a.data / "stage0"
    if a.action == "inventory":
        write_csv(shared / "inventory.csv", inventory(root, real_glob=a.real_glob))
    elif a.action in {"match", "jobs"}:
        rows = read(shared / "inventory.csv")
        base = read(safe_path(root, a.base))
        hindi = [r for r in base if r["language"] == "hi"]
        if a.release and not any(
            r["split"] == "train" and r["generator"] == "freevc24" for r in hindi
        ):
            raise ValueError(
                "Supply release base re-matched with --heldout and no family arguments"
            )
        if a.action == "jobs" or a.addons:
            from .stage0_jobs import addon_jobs, merge

            jobs = addon_jobs(root, hindi, a.microsoft, a.release, rows)
            if a.action == "jobs":
                write_csv(root / f"{work.name}-{a.microsoft}-jobs.csv", jobs)
                return
        selected = match(rows, load_labels(shared, rows), hindi, a.release)
        if a.addons:
            selected = merge(root, selected, jobs, a.release)
        if a.release and not {"freevc24", "chatterboxtts", "gemini"} <= {
            r["generator"] for r in selected if r["split"] == "train"
        }:
            raise ValueError(
                "Stage 5 requires completed add-on jobs and all release families"
            )
        print(json.dumps(counts(selected), indent=2))
        viable(selected)
        write_csv(work / "selection.csv", selected)
        from .diagnostic_panel import mlaad_panel

        diag = mlaad_panel(root / "inventory.jsonl")
        if diag:
            write_csv(work / "mlaad-diagnostic.csv", diag)
    elif a.action == "audit":
        audit_selection(root, work)
    elif a.action == "catalog":
        catalog(root, work)
    else:
        report = json.loads((work / "prep-audit.json").read_text())
        rows = read(work / "catalog.csv")
        eligible = read(work / "eligible.csv")
        if (
            report.get("policy") != POLICY
            or not report["pass"]
            or digest(work / "eligible.csv") != report["eligible_sha256"]
            or digest(work / "selection.csv") != report["selection_sha256"]
            or [[r[k] for k in [*FIELDS, "sha256"] if k != "path"] for r in rows]
            != [[r[k] for k in [*FIELDS, "sha256"] if k != "path"] for r in eligible]
        ):
            raise ValueError("Pack requires audited catalog")
        # An independent packed-byte audit binds Stage 2 to the resulting index.
        catalog_path = root / (work.name + "-catalog.csv")
        write_csv(catalog_path, rows)
        pack(a.data, catalog_path.name, work / "packed")
        report["index_sha256"] = digest(work / "packed" / "index.json")
        report["policy"] = POLICY
        tmp = work / "audit.tmp"
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(work / "audit.json")


if __name__ == "__main__":
    main()
