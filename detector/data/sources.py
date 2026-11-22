"""Dataset adapters. Each yields Record rows and can read any record's bytes.

A locator says where the bytes live without copying or extracting anything:
  file:<path>                  a regular file
  zip:<zip path>::<member>     a member of a zip archive (read in place)
  tar:<tar path>::<offset>:<size>  raw bytes of a member of an uncompressed tar
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
import re
import tarfile
import zipfile

from detector import paths


@dataclass
class Record:
    uid: str            # globally unique: "<source>/<id>"
    source: str         # asvspoof5 | indiefake | in_the_wild | va
    locator: str
    label: int          # 0 genuine, 1 fake
    group: str          # speaker identity, namespaced by source (split unit)
    attack: str         # generator/attack id, "bonafide" for genuine
    role: str           # train | dev | test | unassigned (assigned in the audit)

    def row(self):
        return asdict(self)


# ----------------------------------------------------------------- reading
@lru_cache(maxsize=8)
def _zip(path: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def read_bytes(locator: str) -> bytes:
    kind, rest = locator.split(":", 1)
    if kind == "file":
        return Path(rest).read_bytes()
    if kind == "zip":
        archive, member = rest.split("::", 1)
        return _zip(archive).read(member)
    if kind == "tar":
        archive, span = rest.split("::", 1)
        offset, size = map(int, span.split(":"))
        with open(archive, "rb") as f:
            f.seek(offset)
            return f.read(size)
    raise ValueError(locator)


# ----------------------------------------------------------------- adapters
def asvspoof5():
    """ASVspoof 5 train (A01-A08) and dev (A09-A16), from whichever tars are
    complete. Official partitions are kept: dev attacks are unseen in train."""
    protocol = {}
    for name, role in (("ASVspoof5.train.tsv", "train"), ("ASVspoof5.dev.track_1.tsv", "dev")):
        with open(paths.ASVSPOOF5_PROTOCOLS / name) as f:
            for line in f:
                cols = line.split()
                speaker, utt, attack, key = cols[0], cols[1], cols[7], cols[8]
                protocol[utt] = (speaker, attack, key, role)
    for tar_path in sorted(paths.ASVSPOOF5.glob("flac_[TD]_*.tar")):
        if not Path(f"{tar_path}.ok").exists():
            continue  # still downloading or unverified
        with tarfile.open(tar_path, mode="r:") as archive:
            for m in archive:
                if not m.isfile() or not m.name.endswith(".flac"):
                    continue
                utt = Path(m.name).stem
                speaker, attack, key, role = protocol[utt]
                yield Record(uid=f"asvspoof5/{utt}", source="asvspoof5",
                             locator=f"tar:{tar_path}::{m.offset_data}:{m.size}",
                             label=int(key == "spoof"), group=f"asvspoof5/{speaker}",
                             attack=attack, role=role)


def indiefake():
    """IndieFake: Speaker-*/Bonafides|Deepfakes/*.wav. Split assigned later by speaker."""
    for path in sorted(paths.INDIEFAKE.rglob("*.wav")):
        speaker, category = path.relative_to(paths.INDIEFAKE).parts[:2]
        label = int(category.lower() == "deepfakes")
        yield Record(uid=f"indiefake/{path.relative_to(paths.INDIEFAKE).as_posix()}", source="indiefake",
                     locator=f"file:{path}", label=label, group=f"indiefake/{speaker}",
                     attack="indiefake_fake" if label else "bonafide", role="unassigned")


def in_the_wild():
    """In-the-Wild (Mueller et al. 2022): real and fake public-figure audio. Test only."""
    archive = _zip(str(paths.IN_THE_WILD_ZIP))
    meta_name = next(n for n in archive.namelist() if n.endswith("meta.csv"))
    root = meta_name[: -len("meta.csv")]
    with archive.open(meta_name) as f:
        for row in csv.DictReader(line.decode() for line in f):
            label = row["label"].strip().lower()
            assert label in ("spoof", "bona-fide"), row
            yield Record(uid=f"in_the_wild/{row['file']}", source="in_the_wild",
                         locator=f"zip:{paths.IN_THE_WILD_ZIP}::{root}{row['file']}",
                         label=int(label == "spoof"), group=f"in_the_wild/{row['speaker'].strip()}",
                         attack="itw_fake" if label == "spoof" else "bonafide", role="test")


def va():
    """The 44-file VA development collection. Test only; training copies are removed by the audit."""
    for path in sorted(p for p in paths.VA_DATASET.rglob("*") if p.is_file()):
        category = path.relative_to(paths.VA_DATASET).parts[0]
        label = int(category == "fake")
        stem = path.stem.lower()
        match = re.match(r"([a-z]+)-to-([a-z]+)", stem)
        who = match.group(2) if match else re.sub(r"[-_]?(original|bonafide\d+|\d+)$", "", stem)
        yield Record(uid=f"va/{path.relative_to(paths.VA_DATASET).as_posix()}", source="va",
                     locator=f"file:{path}", label=label, group=f"va/{who}",
                     attack="va_tts" if stem.startswith("tts") else ("va_vc" if label else "bonafide"),
                     role="test")


ADAPTERS = {"asvspoof5": asvspoof5, "indiefake": indiefake, "in_the_wild": in_the_wild, "va": va}


def available():
    """Sources whose data is present and verified."""
    ready = {"indiefake": paths.INDIEFAKE.exists(), "va": paths.VA_DATASET.exists(),
             "in_the_wild": Path(f"{paths.IN_THE_WILD_ZIP}.ok").exists(),
             "asvspoof5": any(Path(f"{t}.ok").exists() for t in paths.ASVSPOOF5.glob("flac_[TD]_*.tar"))}
    return [name for name, ok in ready.items() if ok]
