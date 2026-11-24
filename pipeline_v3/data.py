"""Pack reviewed, development-only records. The index addresses original bytes inside tar."""

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

SPLITS = {"train", "select", "calibrate", "threshold", "validate"}
DENY = {
    "sealed",
    "app_testing_dataset",
    "in_the_wild",
    "itw",
    "va",
    "voxenes",
    "deepfake-eval-2024",
    "private",
    "deepfake-eval",
}
FIELDS = [
    "id",
    "path",
    "split",
    "label",
    "source",
    "pair",
    "language",
    "channel",
    "speaker",
    "voice",
    "parent",
    "text",
    "generator",
    "domain",
]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def safe_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or DENY.intersection(
        s.lower() for s in path.parts
    ):
        raise ValueError("Path outside development pool")
    return path


def check(rows):
    ids, owners, cells = set(), {}, defaultdict(set)
    labels_by_hash = {}
    train_families = {
        r["generator"] for r in rows if r["split"] == "train" and r["label"] == "1"
    }
    for r in rows:
        if any(not r.get(f, "").strip() for f in FIELDS):
            raise ValueError("Missing lineage or schema field")
        if (
            r["split"] not in SPLITS
            or r["label"] not in {"0", "1"}
            or r["id"] in ids
            or r["source"].lower() in DENY
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", r["id"])
        ):
            raise ValueError("Invalid split, label, source or duplicate ID")
        ids.add(r["id"])
        if (sha := r.get("sha256")) and labels_by_hash.setdefault(sha, r["label"]) != r[
            "label"
        ]:
            raise ValueError("Identical audio has conflicting labels")
        if r["label"] == "1" and (
            r["generator"] == "-"
            or r["domain"] not in {"seen", "related", "unseen", "unknown"}
        ):
            raise ValueError("Missing generator family or domain")

        for field in ("speaker", "voice", "parent", "text", "sha256"):
            for key in r.get(field, "").split("|"):
                if not key or key == "-":
                    if field in {"parent", "speaker", "text"} or (
                        field == "voice" and r["label"] == "1"
                    ):
                        raise ValueError(f"Missing {field} identity")
                    continue
                identity = ("person" if field in {"speaker", "voice"} else field, key)
                if owners.setdefault(identity, r["split"]) != r["split"]:
                    raise ValueError(f"{field} leaks across splits")
        if (
            r["label"] == "1"
            and r["domain"] == "unseen"
            and r["generator"] in train_families
        ):
            raise ValueError("Declared unseen generator occurs in train")
        cells[(r["split"], r["pair"], r["language"], r["channel"])].add(r["label"])
    if not rows or any(labels != {"0", "1"} for labels in cells.values()):
        raise ValueError("Unmatched language, channel or source pair")


def pack(data, catalog, out, limit=750_000_000):
    with safe_path(data / "development", catalog).open() as stream:
        rows = sorted(
            csv.DictReader(stream), key=lambda r: (r["split"], r["source"], r["id"])
        )
    check(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError("Choose an unused pack destination")
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        dst = Path(temp) / "pack"
        dst.mkdir()
        shard = None
        previous_group = None
        try:
            for i, r in enumerate(rows):
                payload = safe_path(data / "development", r["path"]).read_bytes()
                group = (r["split"], r["source"])
                if (
                    group != previous_group
                    or not shard
                    or shard.offset + len(payload) > limit
                ):
                    previous_group = group
                    if shard:
                        shard.close()
                    name = f"part-{i:08}.tar"
                    shard = tarfile.open(dst / name, "w", format=tarfile.USTAR_FORMAT)  # noqa: SIM115 (rotated shards close in finally)
                info = tarfile.TarInfo(r["id"])
                info.size = len(payload)
                r.update(
                    shard=name,
                    offset=shard.offset + 512,
                    size=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
                shard.addfile(info, io.BytesIO(payload))
        finally:
            if shard:
                shard.close()
        check(rows)
        (dst / "index.json").write_text(json.dumps(rows))
        files = {p.name: digest(p) for p in dst.iterdir()}
        (dst / "checksums.json").write_text(json.dumps(files))
        dst.rename(out)


def stage(src, dst, splits):
    sums = json.loads((src / "checksums.json").read_text())
    if digest(src / "index.json") != sums["index.json"]:
        raise ValueError("Corrupt index")
    rows = json.loads((src / "index.json").read_text())
    check(rows)
    dst.mkdir(parents=True, exist_ok=True)
    for name in sorted(
        {"index.json", *(r["shard"] for r in rows if r["split"] in splits)}
    ):
        target = safe_path(dst, name)
        if target.exists() and digest(target) == sums[name]:
            continue
        with (
            tempfile.NamedTemporaryFile(dir=dst, delete=False) as tmp,
            safe_path(src, name).open("rb") as f,
        ):
            shutil.copyfileobj(f, tmp)
        tmp = Path(tmp.name)
        if digest(tmp) != sums[name]:
            tmp.unlink()
            raise ValueError(f"Corrupt shard {name}")
        os.replace(tmp, target)


def entries(local, splits):
    rows = json.loads((local / "index.json").read_text())
    check(rows)
    return [r for r in rows if r["split"] in splits]


def payload(local, r):
    with safe_path(local, r["shard"]).open("rb") as f:
        f.seek(r["offset"])
        b = f.read(r["size"])
    if hashlib.sha256(b).hexdigest() != r["sha256"]:
        raise ValueError("Invalid member checksum")
    return b
