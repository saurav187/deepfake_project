"""Resolve a rights-reviewed selection against MLAAD/M-AILABS and Hindi Parquet."""

import csv
import hashlib
import json
import tarfile
from collections import defaultdict
from pathlib import Path

from .data import FIELDS, check, safe_path


def catalog(data, selection, out):
    """The selection adds a reviewed source member to the common lineage columns.

    archive is a development-relative tar/parquet/loose file. member is the exact
    tar member, zero-based Parquet row number or '-' for loose files.
    """
    root = data / "development"
    with safe_path(root, selection).open() as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for r in rows:
        archive = safe_path(root, r["archive"])
        groups[archive].append(r)
        r["path"] = (
            str(archive.relative_to(root))
            if r["member"] == "-"
            else f"selected/{r['id']}.audio"
        )
    check(rows)

    def write(r, b):
        target = safe_path(root, r["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if (
                hashlib.sha256(target.read_bytes()).digest()
                != hashlib.sha256(b).digest()
            ):
                raise ValueError("Selection ID reused with different content")
        else:
            with target.open("xb") as f:
                f.write(b)

    for archive, items in groups.items():
        if all(r["member"] == "-" for r in items):
            if not archive.is_file():
                raise FileNotFoundError(archive)
            continue
        wanted = {r["member"]: r for r in items}
        if len(wanted) != len(items):
            raise ValueError("Duplicate source member")
        if archive.suffix == ".parquet":
            import pyarrow.parquet as pq

            offset = 0
            for batch in pq.ParquetFile(archive).iter_batches(
                batch_size=32, columns=["audio"]
            ):
                for i, audio in enumerate(batch.column(0).to_pylist(), offset):
                    if r := wanted.pop(str(i), None):
                        write(r, audio["bytes"])
                offset += len(batch)
                if not wanted:
                    break
        else:
            with tarfile.open(archive, "r|*") as tar:
                for member in tar:
                    if member.isfile() and (r := wanted.pop(member.name, None)):
                        write(r, tar.extractfile(member).read())
        if wanted:
            raise ValueError(f"Unresolved {len(wanted)} members in {archive.name}")
    path = safe_path(root, out)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def inventory(data, out):
    """Enumerate source metadata without decoding audio. Keep original foreign keys."""
    root = data / "development"
    dest = safe_path(root, out)
    tmp = dest.with_suffix(".tmp")
    with tmp.open("w") as output:

        def emit(source, archive, member, meta):
            print(
                json.dumps(
                    {
                        "source": source,
                        "archive": str(archive.relative_to(root)),
                        "member": member,
                        "metadata": meta,
                    },
                    default=str,
                ),
                file=output,
            )

        for src in ("kathbath", "indicsynth", "indicvoices"):
            for p in sorted((root / src).rglob("train-*.parquet")):
                import pyarrow.parquet as pq

                offset = 0
                pf = pq.ParquetFile(p)
                for batch in pf.iter_batches(batch_size=32):
                    for i, row in enumerate(batch.to_pylist(), offset):
                        audio = row.pop("audio")
                        row["audio_path"] = audio["path"]
                        emit(src, p, str(i), row)
                    offset += len(batch)
        for p in sorted((root / "mlaad").glob("fake/en/*/meta.csv")):
            with p.open() as f:
                for row in csv.DictReader(f, delimiter="|"):
                    audio = p.parent / Path(row["path"]).name
                    if not audio.is_file():
                        raise FileNotFoundError(audio)
                    emit("mlaad", audio, "-", row)
        for p in sorted((root / "mailabs").glob("*.tgz")):
            with tarfile.open(p, "r|gz") as tar:
                for m in tar:
                    if m.isfile() and m.name.lower().endswith(".wav"):
                        emit("mailabs", p, m.name, {"original_file": m.name})
    tmp.replace(dest)
