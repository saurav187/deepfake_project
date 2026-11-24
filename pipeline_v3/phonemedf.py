"""List and join publisher members by LibriSpeech ID. No archive extraction."""

import csv
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from .data import check
from .match import assign, prompt, short

SYSTEMS = {"XTTSv2": "xtts_v2", "MeloTTS": "melotts", "ChatterboxTTS": "chatterboxtts"}
REFS = ("p228", "p230", "p233", "p236", "p243", "p244", "p254", "p258", "p259", "p273")
FOLDS = ("train", "select", "calibrate", "threshold", "validate")
VOICE_FOLD = dict(
    zip(
        sorted(REFS, key=short),
        ["train"] * 5 + ["select"] * 2 + list(FOLDS[2:]),
        strict=True,
    )
)
UID = r"\d+-\d+-\d+"
PINS = {
    "PhonemeDF_real.tar.gz": (9420947621, "71371f453e3d56199f047b0ee16edac1"),
    "PhonemeDF_ChatterboxTTS.tar.gz": (7848724770, "ce8e6a0c2a1456ca201347032c906059"),
    "PhonemeDF_MeloTTS.tar.gz": (7703256782, "e12abaafe46a405e37ed9b24d1f9d76e"),
    "PhonemeDF_XTTSv2.tar.gz": (8193944502, "39ca394061cf89be2c85facf7ba62076"),
    "Documentation_PhonemeDF.pdf": (94221, "ca3f72a479514e332150f64583332330"),
    "PhonemeDF_references.tar.gz": (9219990, "af7a1653df675ff6e8e7d1656761af26"),
}


def archive_name(group):
    if group.startswith("dev-"):
        return "librispeech/" + group + ".tar.gz"
    return f"phonemedf/PhonemeDF_{group}.tar.gz"


def member_id(name):
    match = re.fullmatch(
        f"({UID})(?:[-_][A-Za-z][A-Za-z0-9_-]*)?", Path(name).stem, re.IGNORECASE
    )
    return match[1] if match else None


def read_texts(name, content, texts):
    uid = member_id(name)
    for line in content.decode("utf-8-sig").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if re.fullmatch(UID, parts[0]) and len(parts) == 2:
            key, text = parts
        elif uid:
            key, text = uid, line
        else:
            continue
        if texts.setdefault(key, text) != text:
            raise ValueError(f"Conflicting transcript for {key}")


def inventory(root, listings=None, real_texts=None, real_glob="*.wav"):
    rows = []
    for group in ["real", *SYSTEMS, "dev-clean", "dev-other", "references"]:
        archive = archive_name(group)
        texts, audio = {}, {}

        def visit(name, content=None, group=group, audio=audio, texts=texts):
            path = Path(name)
            if path.suffix.lower() == ".txt" and content is not None:
                read_texts(name, content, texts)
            if path.suffix.lower() not in {".wav", ".flac"}:
                return
            if group == "real" and not path.match(real_glob):
                return
            uid = path.stem.split("_")[0] if group == "references" else member_id(name)
            if not uid or uid in audio:
                raise ValueError(f"Ambiguous or duplicate member {name}")
            audio[uid] = name

        if listings:
            listing = listings / (
                Path(archive).name.removesuffix(".tar.gz") + ".listing.txt"
            )
            if group == "real" and not listing.exists():
                continue
            for line in listing.read_text().splitlines():
                if line.startswith("-"):
                    visit(line.split(maxsplit=5)[5])
            if group == "real" and real_texts:
                for p in real_texts.rglob("*.txt"):
                    read_texts(p.name, p.read_bytes(), texts)
        else:
            with tarfile.open(root / archive, "r|gz") as tar:
                for m in tar:
                    if m.isfile():
                        is_text = m.name.lower().endswith(".txt") and (
                            group == "real" or m.name.endswith(".trans.txt")
                        )
                        visit(m.name, tar.extractfile(m).read() if is_text else None)
        if not audio or (group == "references" and set(audio) != set(REFS)):
            raise ValueError(f"Incomplete archive {archive}")
        for uid, member in audio.items():
            reader, _, chapter = uid.partition("-")
            rows.append(
                {
                    "group": group,
                    "uid": uid,
                    "reader": reader,
                    "chapter": chapter.partition("-")[0],
                    "archive": archive,
                    "member": member,
                    "transcript": texts.get(uid, ""),
                }
            )
    sets = {g: {r["uid"] for r in rows if r["group"] == g} for g in ["real", *SYSTEMS]}
    if any(s != sets["XTTSv2"] for g, s in sets.items() if g != "real" or s):
        raise ValueError("Unequal utterance sets across PhonemeDF archives")
    dev = [
        {r["reader"] for r in rows if r["group"] == g}
        for g in ("dev-clean", "dev-other")
    ]
    if dev[0] & dev[1] or (dev[0] | dev[1]) & {
        uid.split("-")[0] for uid in sets["XTTSv2"]
    }:
        raise ValueError("LibriSpeech reader overlap")
    if not listings and any(
        not r["transcript"]
        for r in rows
        if r["group"] in {"real", "dev-clean", "dev-other"}
    ):
        raise ValueError("Missing real transcript; inspect archive naming")
    return rows


def write_csv(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def match(rows, labels, hindi, release=False):
    real = {r["uid"]: r for r in rows if r["group"] == "real"}
    kept = {r["id"]: r for r in hindi}
    owners = {r["text"]: r["split"] for r in hindi}
    # Fix prompt ownership before voice inference so admitting a family for the
    # release cannot move a duplicated sentence into a different fold.
    for r in sorted(real.values(), key=lambda r: short(r["uid"])):
        if r["transcript"]:
            owners.setdefault(
                prompt(r["transcript"]), assign("librispeech/" + r["reader"])
            )
    for r in sorted(
        (r for r in rows if r["group"] in SYSTEMS), key=lambda r: short(r["member"])
    ):
        uid, group = r["uid"], r["group"]
        original = real.get(uid)
        if not original or not original["transcript"]:
            continue
        voice = labels.get((group, uid), "unknown")
        split = assign("librispeech/" + r["reader"])
        if voice == "builtin-cohort" and group == "MeloTTS":
            target_fold = "train"
        else:
            target_fold = VOICE_FOLD.get(voice)
        if target_fold != split or (
            split == "train" and group == "ChatterboxTTS" and not release
        ):
            continue
        key = prompt(original["transcript"])
        if owners.setdefault(key, split) != split:
            continue
        for item, fake in [(original, False), (r, True)]:
            rid = short(item["archive"] + "/" + item["member"])
            kept[rid] = {
                "id": rid,
                "archive": item["archive"],
                "member": item["member"],
                "path": item["archive"],
                "split": split,
                "label": str(int(fake)),
                "source": "phonemedf",
                "pair": "phonemedf",
                "language": "en",
                "channel": "clean",
                "speaker": "librispeech/" + r["reader"],
                "voice": ("sv/" + voice if fake else "-"),
                "parent": "librispeech/" + uid,
                "text": key,
                "generator": SYSTEMS[group] if fake else "-",
                "domain": (
                    "unseen" if group == "ChatterboxTTS" and not release else "seen"
                )
                if fake
                else "real",
            }
    result = list(kept.values())
    if result:
        check(result)
    return result


def counts(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[r["language"], r["split"], r["generator"], r["label"]].append(r)
    return {
        "/".join(k): [
            len(v),
            len({r["speaker"] for r in v}),
            len({r["voice"] for r in v} - {"-", "sv/builtin-cohort"}),
        ]
        for k, v in sorted(cells.items())
    }


def proof(rows, hindi):
    print("members", Counter(r["group"] for r in rows))
    print(
        "resolved real transcripts",
        sum(bool(r["transcript"]) for r in rows if r["group"] == "real"),
    )
    for split in FOLDS:
        eligible = [
            r
            for r in rows
            if r["group"] == "XTTSv2" and assign("librispeech/" + r["reader"]) == split
        ]
        print(split, len(eligible), len({r["reader"] for r in eligible}))
    texts = defaultdict(set)
    for r in rows:
        if r["group"] == "real" and r["transcript"]:
            texts[prompt(r["transcript"])].add(assign("librispeech/" + r["reader"]))
    print(
        "distinct normalized transcripts",
        len(texts),
        "cross-fold repeats",
        sum(len(v) > 1 for v in texts.values()),
    )
    print("accepted without inferred labels", counts(match(rows, {}, hindi)))
