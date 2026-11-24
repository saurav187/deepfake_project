"""Construct candidate matched pairs from publisher metadata. Omit unresolved lineage."""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .data import FIELDS, check, safe_path


def norm(value):
    return "" if value is None else str(value).strip().casefold()


def short(value):
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def prompt(value):
    text = " ".join(norm(value).split())
    return short(text) if text else ""


def assign(person):
    u = int(short(person)[:8], 16) % 100
    for threshold, split in (
        (50, "train"),
        (68, "select"),
        (81, "calibrate"),
        (91, "threshold"),
        (100, "validate"),
    ):
        if u < threshold:
            return split


def match(data, inventory, selection, heldout, cap=2500):
    root = data / "development"
    with safe_path(root, inventory).open() as f:
        records = [json.loads(line) for line in f]
    reals = defaultdict(list)
    rejected = Counter()
    for r in records:
        m = r["metadata"]
        src = r["source"]
        if src == "mailabs":
            parts = Path(r["member"]).parts
            pos = next(
                (i for i, s in enumerate(parts) if s in {"male", "female", "mix"}), None
            )
            if pos is None or pos + 1 >= len(parts) or parts[pos] == "mix":
                continue
            person = "mailabs/" + norm(parts[pos + 1])
            filename = parts[-1]
            text = ""
        elif src == "kathbath":
            filename = Path(
                str(m.get("audio_path") or m.get("audio_filepath") or "")
            ).name
            # Kathbath IDs include an utterance and a numeric speaker component.
            parsed = re.search(r"-(\d+)-[mf](?:\.|$)", filename, re.IGNORECASE)
            sp = m.get("speaker_id") or m.get("speaker")
            if not sp and not parsed:
                continue
            person = "kathbath/" + str(sp or parsed[1]).removesuffix(".0")
            text = prompt(m.get("text") or m.get("transcript"))
        else:
            continue
        if filename:
            reals[src, Path(filename).stem].append((r, person, text))
    ownership = {}
    kept = {}
    counts = Counter()
    heldout = {norm(v) for v in heldout}
    for fake in sorted(
        (r for r in records if r["source"] in {"mlaad", "indicsynth"}),
        key=lambda r: short(r["archive"] + "/" + r["member"]),
    ):
        m = fake["metadata"]
        if fake["source"] == "mlaad":
            src, lang, family = "mailabs", "en", Path(fake["archive"]).parent.name
            parent = m.get("original_file")
            identities = [m.get("reference_speaker")]
            transcript = m.get("transcript")
            if norm(m.get("is_original_language")) in {"false", "0", "no"}:
                rejected["translated text"] += 1
                continue
        else:
            src, lang, family = "kathbath", "hi", m.get("Generative Model", "")
            parent = m.get("Source Reference Audio") or m.get("Target Reference Audio")
            identities = [m.get("Target Speaker ID")]
            transcript = m.get("TTS Transcript")
        ref = reals.get((src, Path(str(parent or "")).stem), [])
        if (
            len(ref) != 1
            or norm(identities[0]) in {"", "nan", "none", "unknown", "null"}
            or not family
        ):
            rejected["unresolved parent or voice"] += 1
            continue
        real, person, realtext = ref[0]
        if lang == "en":
            # Reference can be a reader identifier or an original audio path.
            voice = norm(identities[0])
            rv = reals.get((src, Path(voice).stem), [])
            if len(rv) == 1:
                voice = rv[0][1]
            elif "/" in voice or voice.endswith((".wav", ".flac")):
                rejected["unresolved reference audio"] += 1
                continue
            else:
                voice = src + "/" + voice
        else:
            voice = src + "/" + str(identities[0]).removesuffix(".0")
        split = assign(person)
        if assign(voice) != split or (split == "train" and norm(family) in heldout):
            rejected["reserved speaker or family"] += 1
            continue
        realid = short(real["archive"] + "/" + real["member"])
        fakeid = short(fake["archive"] + "/" + fake["member"])
        parentkey = src + "/" + realid
        textid = prompt(transcript) or realtext
        if not textid or any(
            ownership.get(t, split) != split for t in {textid, realtext} if t
        ):
            rejected["missing or crossing prompt"] += 1
            continue
        cell = (lang, split, norm(family))
        if counts[cell] >= cap:
            continue
        # A reused real must keep all text keys of its associated utterance family.
        realkey = realtext or textid
        if realid in kept and kept[realid]["text"] != realkey:
            rejected["inconsistent original transcript"] += 1
            continue
        for t in {textid, realkey}:
            ownership[t] = split
        for r, uid, lab, who, txt, gen, dom in (
            (real, realid, "0", "-", realkey, "-", "real"),
            (
                fake,
                fakeid,
                "1",
                voice,
                textid,
                norm(family),
                "unseen" if norm(family) in heldout else "seen",
            ),
        ):
            kept[uid] = {
                "id": uid,
                "archive": r["archive"],
                "member": r["member"],
                "path": r["archive"],
                "split": split,
                "label": lab,
                "source": r["source"],
                "pair": src,
                "language": lang,
                "channel": "clean",
                "speaker": person,
                "voice": who,
                "parent": parentkey,
                "text": txt,
                "generator": gen,
                "domain": dom,
            }
        counts[cell] += 1
    check(list(kept.values()))
    dest = safe_path(root, selection)
    with dest.open("x", newline="") as f:
        w = csv.DictWriter(f, ["archive", "member", *FIELDS])
        w.writeheader()
        w.writerows(kept.values())
    print(
        json.dumps(
            {
                "selected": len(kept),
                "excluded": rejected,
                "counts": {"/".join(k): v for k, v in counts.items()},
            },
            indent=2,
        )
    )
