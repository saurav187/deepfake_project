"""Small optional commercial panel. Bind each stock rendition to a development parent."""

import json
from collections import defaultdict

from .data import check
from .english import VOICES
from .generate import generated_path
from .match import assign, prompt, short
from .phonemedf import FOLDS


def addon_jobs(root, hindi, microsoft, release, inventory_rows):
    # Reserve PhonemeDF prompts before selecting any commercial jobs. Allocate
    # dev readers once, independent of the requested service or release stage.
    used = {prompt(r["transcript"]) for r in inventory_rows if r["group"] == "real"}
    used.update(r["text"] for r in hindi)
    buckets = defaultdict(list)
    for r in sorted(
        (r for r in inventory_rows if r["group"] in {"dev-clean", "dev-other"}),
        key=lambda r: short(r["uid"]),
    ):
        text = r["transcript"]
        key = prompt(text)
        if key in used or not 6 <= len(text.split()) <= 45:
            continue
        used.add(key)
        speaker = "librispeech/" + r["reader"]
        split = assign(speaker)
        engine = (
            "gemini"
            if int(short(r["uid"])[0], 16) % 2 and (release or split != "train")
            else microsoft
        )
        pool = VOICES["gemini" if engine == "gemini" else "azure"][FOLDS.index(split)]
        voice = pool[int(short(speaker)[0], 16) % len(pool)]
        real_id = short(r["archive"] + "/" + r["member"])
        buckets["en", split, engine].append(
            {
                "id": short(real_id + voice + engine),
                "text": text,
                "engine": engine,
                "model": {
                    "gemini": "gemini-3.1-flash-tts-preview",
                    "edge": "edge-tts",
                    "azure": "azure-neural",
                }[engine],
                "voice": voice,
                "speaker": speaker,
                "language": "en",
                "split": split,
                "seconds": str(round(len(text.split()) / 2.5, 2)),
                "archive": r["archive"],
                "member": r["member"],
                "real_id": real_id,
                "parent": "librispeech/" + r["uid"],
                "pair": "libri-stock",
            }
        )
    originals = {(r["archive"], r["member"]): r for r in hindi if r["label"] == "0"}
    with (root / "inventory.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            real = originals.get((r["archive"], r["member"]))
            if not real:
                continue
            text = r["metadata"].get("text", "")
            if not text or prompt(text) != real["text"]:
                continue
            split = real["split"]
            # Only two Hindi stock Microsoft voices: give each exactly one fold.
            ms = {"train": "hi-IN-SwaraNeural", "select": "hi-IN-MadhurNeural"}
            gm = {
                "train": "Iapetus",
                "select": "Gacrux",
                "calibrate": "Despina",
                "threshold": "Enceladus",
                "validate": "Umbriel",
            }
            for engine, voice in [(microsoft, ms.get(split)), ("gemini", gm[split])]:
                if not voice or (
                    engine == "gemini" and split == "train" and not release
                ):
                    continue
                job = {
                    "id": short(real["id"] + voice + engine),
                    "text": text,
                    "engine": engine,
                    "model": "gemini-3.1-flash-tts-preview"
                    if engine == "gemini"
                    else ("edge-tts" if engine == "edge" else "azure-neural"),
                    "voice": voice,
                    "speaker": real["speaker"],
                    "language": "hi",
                    "split": split,
                    "seconds": str(r["metadata"]["duration"]),
                    "archive": real["archive"],
                    "member": real["member"],
                    "real_id": real["id"],
                    "parent": real["parent"],
                    "pair": real["pair"],
                }
                buckets["hi", split, engine].append(job)
    return [
        j
        for key in sorted(buckets)
        for j in sorted(buckets[key], key=lambda j: short(j["id"]))[:24]
    ]


def merge(root, rows, jobs, release):
    result = {r["id"]: r for r in rows}
    for j in jobs:
        path = generated_path(j)
        if not (root / path).is_file() or not (root / path).stat().st_size:
            raise ValueError(f"Missing {path}")
        common = {
            "split": j["split"],
            "pair": j["pair"],
            "language": j["language"],
            "channel": "clean",
            "speaker": j["speaker"],
            "parent": j["parent"],
            "text": prompt(j["text"]),
        }
        real = j["real_id"]
        result.setdefault(
            real,
            dict(
                common,
                id=real,
                archive=j["archive"],
                member=j["member"],
                path=j["archive"],
                source="librispeech",
                label="0",
                voice="-",
                generator="-",
                domain="real",
            ),
        )
        uid = short(str(path))
        result[uid] = dict(
            common,
            id=uid,
            archive=str(path),
            member="-",
            path=str(path),
            source=j["engine"],
            label="1",
            voice=j["engine"] + "/" + j["voice"].casefold(),
            generator=j["engine"],
            domain="unseen" if j["engine"] == "gemini" and not release else "seen",
        )
    check(list(result.values()))
    return list(result.values())
