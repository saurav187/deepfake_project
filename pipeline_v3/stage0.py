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


TRAIN_FAMILIES = {"xtts_v2", "melotts"}
# Owner-approved separate experiment (25 Nov 2026, option B): English XTTS-only train cell,
# Chatterbox still fully held out. The default two-generator recipe above is unchanged.
XTTS_ONLY_TRAIN = {"xtts_v2"}


def viable(rows, train_need=TRAIN_FAMILIES):
    check(rows)
    if train_need == XTTS_ONLY_TRAIN:
        for r in rows:
            allowed = train_need if r["split"] == "train" else {"xtts_v2", "chatterboxtts"}
            if (r["pair"] != "phonemedf" or r["language"] != "en"
                    or (r["label"] == "1" and r["generator"] not in allowed)):
                raise ValueError("Unexpected row in English XTTS-only experiment")
    for fold in ("train", "select", "calibrate", "threshold", "validate"):
        en = [
            r
            for r in rows
            if r["language"] == "en" and r["split"] == fold and r["pair"] == "phonemedf"
        ]
        if len({r["speaker"] for r in en}) < (20 if fold == "train" else 5):
            raise ValueError("Insufficient PhonemeDF readers in " + fold)
        need = set(train_need) if fold == "train" else {"chatterboxtts"}
        if not need <= {r["generator"] for r in en}:
            raise ValueError("Missing English generator cell in " + fold)


# Owner-authorized smaller English pilot (25 Nov 2026): fake quotas per fold and family,
# fixed before any detector score exists. Genuine rows are exactly the parents of kept fakes.
SUBSET_QUOTAS = {
    "train": {"xtts_v2": 1500, "melotts": 1500},
    "select": {"xtts_v2": 600, "chatterboxtts": 600},
    "calibrate": {"xtts_v2": 150, "chatterboxtts": 150},
    "threshold": {"xtts_v2": 150, "chatterboxtts": 150},
    "validate": {"xtts_v2": 150, "chatterboxtts": 150},
}


def _h(text):
    return hashlib.sha256(("subset-v1/" + text).encode()).hexdigest()


def subset(rows, quotas=SUBSET_QUOTAS):
    """Deterministic reader-round-robin fake quotas; keeps matched parents, never adds rows."""
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["language"] == "en" and r["label"] == "1":
            cells[r["split"], r["generator"]][r["speaker"]].append(r)
    kept = set()
    for (split, gen), readers in cells.items():
        quota = quotas.get(split, {}).get(gen, 0)
        queues = [
            sorted(v, key=lambda r: _h(r["id"]))
            for _, v in sorted(readers.items(), key=lambda kv: _h(kv[0]))
        ]
        taken = 0
        while taken < quota and any(queues):
            for q in queues:
                if q and taken < quota:
                    kept.add(q.pop(0)["id"])
                    taken += 1
    parents = {r["parent"] for r in rows if r["id"] in kept}
    return [
        r
        for r in rows
        if r["language"] == "en"
        and (r["id"] in kept or (r["label"] == "0" and r["parent"] in parents))
    ]


def audit_selection(root, work, languages=("en", "hi"), train_need=TRAIN_FAMILIES):
    selection_sha256 = digest(work / "selection.csv")
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
    viable(kept, train_need)
    idx = [r for r in kept if r["id"] in features]
    report = panels(idx, [features[r["id"]] for r in idx], languages)
    if digest(work / "selection.csv") != selection_sha256:
        raise ValueError("Selection changed during audit")
    report.update(
        selection_sha256=selection_sha256,
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


def catalog(root, work, languages=("en", "hi")):
    report = json.loads((work / "prep-audit.json").read_text())
    if (
        report.get("policy") != POLICY
        or report.get("languages_audited") != sorted(languages)
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


def load_labels(work, rows, inventory_path=None):
    if inventory_path is None:
        inventory_path = work / "inventory.csv"
    expected = {(r["group"], r["uid"]) for r in rows if r["group"] in SYSTEMS}
    if (
        digest(inventory_path)
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
    p.add_argument(
        "--languages",
        nargs="+",
        choices=["en", "hi"],
        default=["en", "hi"],
        help="'--languages en' is the English-only path; its products live in <stage>-en/",
    )
    p.add_argument(
        "--subset",
        action="store_true",
        help="explicitly labelled smaller English pilot (requires --languages en); "
        "products live in <stage>-en-subset/",
    )
    p.add_argument(
        "--voices",
        choices=["wavlm", "ecapa"],
        default="wavlm",
        help="voice-label source: stage0/voices.csv (WavLM) or stage0-ecapa/voices.csv",
    )
    p.add_argument(
        "--xtts-only",
        action="store_true",
        help="separately named English experiment: XTTS-v2 is the only train family, "
        "MeloTTS rows are dropped, Chatterbox stays held out; products in <stage>-...-xtts/",
    )
    a = p.parse_args()
    languages = sorted(set(a.languages))
    if a.subset and (languages != ["en"] or a.release or a.addons):
        raise SystemExit("--subset is only for the English-only Stage 1/2 pilot")
    if a.xtts_only and (languages != ["en"] or a.voices != "ecapa" or a.release or a.addons or a.subset):
        raise SystemExit("--xtts-only requires --languages en --voices ecapa (Stage 1/2 only)")
    train_need = XTTS_ONLY_TRAIN if a.xtts_only else TRAIN_FAMILIES
    root = a.data / "development"
    work = a.data / ("stage5" if a.release else "stage3" if a.addons else "stage0")
    if languages == ["en"]:
        work = work.with_name(work.name + ("-en-subset" if a.subset else "-en"))
    if a.voices == "ecapa":
        work = work.with_name(work.name + "-ecapa" + ("-xtts" if a.xtts_only else ""))
    elif languages != ["en", "hi"]:
        raise SystemExit("Hindi-only runs are not supported; English is always required")
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
        hindi = [r for r in base if r["language"] == "hi" and "hi" in languages]
        if a.release and "hi" in languages and not any(
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
        voice_dir = shared if a.voices == "wavlm" else a.data / "stage0-ecapa"
        selected = match(rows, load_labels(voice_dir, rows, shared / "inventory.csv"), hindi, a.release)
        if a.addons:
            selected = merge(root, selected, jobs, a.release)
        if a.release and not {"freevc24", "chatterboxtts", "gemini"} <= {
            r["generator"] for r in selected if r["split"] == "train"
        }:
            raise ValueError(
                "Stage 5 requires completed add-on jobs and all release families"
            )
        if a.xtts_only:
            selected = [r for r in selected if r["generator"] != "melotts"]
            fakes = {r["parent"] for r in selected if r["label"] == "1" and r["language"] == "en"}
            selected = [r for r in selected if r["label"] == "1" or r["parent"] in fakes]
            (work / "experiment.json").write_text(json.dumps({
                "kind": "ENGLISH XTTS-ONLY EXPERIMENT (owner option B, 25 Nov 2026)",
                "train_families": sorted(XTTS_ONLY_TRAIN),
                "held_out": ["chatterboxtts"],
                "excluded": ["melotts"],
                "voices": "stage0-ecapa/voices.csv, frozen rule ecapa-reference-floor-v1",
            }, indent=2))
        if a.subset:
            full = len(selected)
            selected = subset(selected)
            (work / "subset.json").write_text(
                json.dumps(
                    {
                        "kind": "SMALLER ENGLISH PILOT SUBSET (not the full run)",
                        "quotas": SUBSET_QUOTAS,
                        "rule": "subset-v1 sha256 reader round-robin; parents of kept fakes",
                        "full_selection_rows": full,
                        "subset_rows": len(selected),
                    },
                    indent=2,
                )
            )
        print(json.dumps(counts(selected), indent=2))
        viable(selected, train_need)
        write_csv(work / "selection.csv", selected)
        from .diagnostic_panel import mlaad_panel

        diag = mlaad_panel(root / "inventory.jsonl")
        if diag:
            write_csv(work / "mlaad-diagnostic.csv", diag)
    elif a.action == "audit":
        audit_selection(root, work, languages, train_need)
    elif a.action == "catalog":
        catalog(root, work, languages)
    else:
        report = json.loads((work / "prep-audit.json").read_text())
        rows = read(work / "catalog.csv")
        eligible = read(work / "eligible.csv")
        if (
            report.get("policy") != POLICY
            or report.get("languages_audited") != languages
            or not report["pass"]
            or digest(work / "eligible.csv") != report["eligible_sha256"]
            or digest(work / "selection.csv") != report["selection_sha256"]
            or [[r[k] for k in [*FIELDS, "sha256"] if k != "path"] for r in rows]
            != [[r[k] for k in [*FIELDS, "sha256"] if k != "path"] for r in eligible]
        ):
            raise ValueError("Pack requires audited catalog")
        viable(eligible, train_need)
        if any(r["path"] != f"{work.name}/bytes/{r['id']}.audio" for r in rows):
            raise ValueError("Catalog path belongs to a different experiment")
        # The catalog hashes verified by pack bind the preparation audit to this index.
        catalog_path = root / (work.name + "-catalog.csv")
        write_csv(catalog_path, rows)
        packed = work / "packed"
        if packed.exists():
            # A disconnect can land between publication of packed/ and audit.json.
            # Reuse only an index describing precisely this audited catalog.
            index = json.loads((packed / "index.json").read_text())
            ordered = sorted(rows, key=lambda r: (r["split"], r["source"], r["id"]))
            if [{k: r[k] for k in [*FIELDS, "sha256"]} for r in index] != ordered:
                raise ValueError("Existing pack differs from the audited catalog")
            checksums = json.loads((packed / "checksums.json").read_text())
            if digest(packed / "index.json") != checksums["index.json"]:
                raise ValueError("Existing pack has a corrupt index")
        else:
            pack(a.data, catalog_path.name, packed)
        report["index_sha256"] = digest(work / "packed" / "index.json")
        report["policy"] = POLICY
        tmp = work / "audit.tmp"
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(work / "audit.json")


if __name__ == "__main__":
    main()
