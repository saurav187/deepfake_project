"""Independent English calibration / evaluation manifests for the modern-TTS 70%-at-15% target.

The three-reader MLAAD panel (teacher_eval.panel_rows, 250 per family) is DEVELOPMENT: it has
been viewed repeatedly. This module builds disjoint manifests and scores them with frozen
models only. No model, weight, normaliser or cutoff is fitted here.

  build   manifests from metadata only (no audio)                       -> <out>/manifest-*.csv
  score   frozen release v1 (paired 4 s crop view) + frozen Teacher V1   -> <out>/scores/*.json
  report  apply the externally frozen candidate cutoff                   -> <out>/result.json

Calibration (cutoff source, owned by the candidate's freezer):
  genuine  libri-dev: the 841 LibriSpeech dev-clean/dev-other clips of teacher-eval-clips.csv
           (73 readers). Never used to fit any of these models; teacher V1 aggregates were seen.
  fakes    unused MLAAD rows with en_US parents (new sentences of the development books).
           Descriptive only: the cutoff is set on calibration genuine audio alone, so these
           fakes share stock voices with evaluation and development by design.
Evaluation (scored only after the candidate and its cutoff are frozen):
  fakes    MLAAD rows with en_UK parents (Elizabeth Klett books, none in dev or calibration).
           Every ElevenLabs/Gemini row is kept (owner priority); their stock voices also occur
           in development and in the descriptive calibration fakes.
  genuine  primary: RVCBench ground-truth recordings of LibriTTS test-clean speakers
           (upstream_role == upstream_eval:test-clean; never in V1 readers, MLAAD readers,
           PhonemeDF reference voices or any training arm). Secondary, reported separately:
           VCC 2018 genuine source/target recordings (DAPS speakers).
  proxy    OpenAudio S1-mini (Fish family, RVCBench) fakes on the same test-clean speakers.
           A diagnostic proxy, never counted as MLAAD FishTTS.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

FAMILIES = ["Chatterbox", "ElevenLabs-v3", "FishTTS", "Gemini-3.1-Flash-TTS"]
MLAAD_REVISION = "30c3dec763fa4803f11c2df1557d3b0828395366"
MLAAD_DIR = "development/mlaad/fake/en"
DEV_PER_FAMILY = 250  # teacher_eval.MLAAD_PER_FAMILY
TEACHER_CSV = "stage0-en-ecapa-xtts-norm-exploratory/runs/teacher-eval-clips.csv"
V1_CATALOG = "development/stage0-en-ecapa-xtts-norm-exploratory-catalog.csv"
TEACHER_RUNS = "stage0-en-ecapa-xtts-norm-exploratory/runs"
INTAKE = "development/modern-intake/v1"
VCC_ZIPS = ("vcc2018_database_evaluation.zip", "vcc2018_database_reference.zip")
PROXY_MODELS = {"OpenAudio S1-mini": "Fish-family proxy (OpenAudio S1-mini via RVCBench), not MLAAD FishTTS"}
FIELDS = ["id", "split", "stratum", "label", "family", "source", "locator", "speaker", "voice",
          "text_key", "parent", "book"]


def text_key(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_mlaad(root):
    meta = {}
    for fam in FAMILIES:
        with (root / MLAAD_DIR / fam / "meta.csv").open() as f:
            for r in csv.DictReader(f, delimiter="|"):
                key = "mlaad/fake/en/" + r["path"].split("/fake/en/")[1]
                meta[key] = dict(r, family=fam)
    return meta


def dev_sample(root):
    """Exact reproduction of teacher_eval.panel_rows' MLAAD sample (the development panel)."""
    by = defaultdict(list)
    with (root / "stage0-en-ecapa-xtts" / "mlaad-diagnostic.csv").open() as f:
        for r in csv.DictReader(f):
            by[r["family"]].append(r)
    out = set()
    for v in by.values():
        v = sorted(v, key=lambda r: hashlib.sha256(r["fake_member"].encode() + r["fake_archive"].encode()).hexdigest())
        out |= {r["fake_archive"] for r in v[:DEV_PER_FAMILY]}
    return out


def build(root, out, pairs_path):
    meta = read_mlaad(root)
    sample = dev_sample(root)
    with (root / TEACHER_CSV).open() as f:
        te = [r for r in csv.DictReader(f) if r["channel"] == "clean"]
    dev_ids = {r["id"].removeprefix("mlaad/") for r in te if r["panel"] == "mlaad" and r["label"] == "1"}
    if not dev_ids <= sample:
        raise ValueError("Development panel no longer reproduces")
    dev_parents = {meta[k]["original_file"] for k in sample}
    dev_texts = {text_key(meta[k]["transcript"]) for k in sample}
    rows, dropped = [], Counter()
    for k, r in sorted(meta.items()):
        fam = r["family"]
        if k in sample:
            dropped[f"{fam}|development_panel_sample"] += 1
            continue
        tk = text_key(r["transcript"])
        if r["original_file"] in dev_parents or tk in dev_texts:
            dropped[f"{fam}|development_text_or_parent"] += 1
            continue
        region = r["original_file"].split("/")[0]
        split = {"en_US": "calibration", "en_UK": "evaluation"}[region]
        voice = r.get("reference_speaker", "-") or "-"
        parts = r["original_file"].split("/")
        rows.append(dict(id=k, split=split, stratum="mlaad", label=1, family=fam, source="mlaad",
                         locator=f"{MLAAD_DIR}/{fam}/{Path(k).name}", speaker="tts/" + fam,
                         voice=voice if voice != "-" else f"{fam}:unstated", text_key=tk,
                         parent="mailabs/" + r["original_file"], book=parts[4], _text=r["transcript"]))
    # Calibration genuine: the exact libri-dev rows already scored by the teacher.
    for r in te:
        if r["panel"].startswith("libri-dev"):
            rows.append(dict(id=r["id"], split="calibration", stratum="libri-dev", label=0, family="genuine",
                             source="libri-dev", locator=r["id"], speaker=r["speaker"], voice="",
                             text_key="", parent=r["id"], book=""))
    # Evaluation genuine (primary) + Fish-family proxy from RVCBench test-clean speakers.
    pairs = [json.loads(line) for line in Path(pairs_path).read_text().splitlines() if line.strip()]
    seen = set()
    for p in pairs:
        if p["upstream_role"] != "upstream_eval:test-clean" or p["corpus"] != "Libritts":
            continue
        if p["genuine_eligible"] and p["genuine"] not in seen:
            seen.add(p["genuine"])
            rows.append(dict(id="rvcbench/" + p["genuine"], split="evaluation", stratum="libritts-test-clean",
                             label=0, family="genuine", source="rvcbench", locator=p["genuine"],
                             speaker=p["speaker"], voice="", text_key=text_key(p["text"]),
                             parent=p["genuine"], book="", _text=p["text"]))
        if p["model"] in PROXY_MODELS and p["fake_eligible"]:
            rows.append(dict(id="rvcbench/" + p["fake"], split="evaluation", stratum="proxy",
                             label=1, family="proxy/" + p["model"], source="rvcbench", locator=p["fake"],
                             speaker=p["speaker"], voice="clone:" + p["speaker"], text_key=text_key(p["text"]),
                             parent=p["genuine"], book="", _text=p["text"]))
    # Evaluation genuine (secondary): VCC 2018 genuine speech, all members of the two zips.
    for z in VCC_ZIPS:
        with zipfile.ZipFile(root / "development/vcc2018" / z) as zf:
            for m in sorted(zf.namelist()):
                if m.endswith(".wav"):
                    spk = Path(m).parent.name
                    rows.append(dict(id=f"vcc2018/{z}/{m}", split="evaluation", stratum="vcc2018",
                                     label=0, family="genuine", source="vcc2018", locator=f"{z}::{m}",
                                     speaker="vcc2018/" + spk, voice="", text_key="",
                                     parent=f"vcc2018/{spk}/{Path(m).stem}", book=""))
    ids = Counter(r["id"] for r in rows)
    if max(ids.values()) > 1:
        raise ValueError("Duplicate manifest ids")
    cal_t = {r["text_key"] for r in rows if r["split"] == "calibration" and r["text_key"]}
    ev_t = {r["text_key"] for r in rows if r["split"] == "evaluation" and r["text_key"]}
    cal_v = {r["voice"] for r in rows if r["split"] == "calibration" and r["voice"]}
    ev_v = {r["voice"] for r in rows if r["split"] == "evaluation" and r["voice"]}
    cal_s = {r["speaker"] for r in rows if r["split"] == "calibration" and r["label"] == 0}
    ev_s = {r["speaker"] for r in rows if r["split"] == "evaluation" and r["label"] == 0}
    checks = {
        "text_overlap_calibration_evaluation": len(cal_t & ev_t),
        "text_overlap_evaluation_development": len(ev_t & dev_texts),
        "named_voice_overlap_calibration_evaluation_descriptive_fakes": len({v for v in cal_v & ev_v if not v.endswith(":unstated")}),
        "unstated_voice_families_shared": sorted(v for v in cal_v & ev_v if v.endswith(":unstated")),
        "genuine_speaker_overlap_calibration_evaluation": sorted(cal_s & ev_s),
        "genuine_speaker_overlap_evaluation_development": sorted(ev_s & {r["speaker"] for r in te if r["panel"] == "mlaad"}),
    }
    per_original = Counter((r["split"], r["family"], r["parent"]) for r in rows if r["source"] == "mlaad")
    checks["mlaad_originals_repeated_within_family"] = sum(v > 1 for v in per_original.values())
    # Teacher V1 pack texts (match.prompt keys) vs evaluation texts (v2l.text_key, same space).
    from .v2l import text_key as v1_key

    with (root / V1_CATALOG).open() as f:
        v1_texts = {r["text"] for r in csv.DictReader(f) if r["text"]}
    ev_raw = {r["id"]: r["_text"] for r in rows if r["split"] == "evaluation" and r.get("_text")}
    hits = sorted(i for i, t in ev_raw.items() if v1_key(t) in v1_texts)
    checks["evaluation_text_in_teacher_v1_pack"] = {"checked": len(ev_raw), "hits": len(hits), "ids": hits[:50]}
    checks["release_v1_overlap"] = ("exact decoded-PCM match against the release's 138,180 train/dev hashes is "
                                    "checked per clip at scoring for MLAAD and VCC (raw bytes); RVCBench clips "
                                    "are cached crops, so exact overlap is unverifiable there. ASVspoof 5 speakers "
                                    "are anonymised: LibriVox reader overlap is unverifiable.")
    for r in rows:
        r.pop("_text", None)
    if checks["text_overlap_calibration_evaluation"] \
            or checks["genuine_speaker_overlap_calibration_evaluation"] or checks["text_overlap_evaluation_development"]:
        raise ValueError(f"Split leakage: {checks}")
    out.mkdir(parents=True, exist_ok=True)
    for split in ("calibration", "evaluation"):
        p = out / f"manifest-{split}.csv"
        new = io.StringIO()
        w = csv.DictWriter(new, FIELDS)
        w.writeheader()
        w.writerows(r for r in rows if r["split"] == split)
        if p.exists() and p.read_bytes() != new.getvalue().encode():
            raise ValueError(f"{p.name} exists with different membership; manifests are frozen")
        p.write_bytes(new.getvalue().encode())
    counts = defaultdict(lambda: {"clips": 0, "speakers_or_voices": set(), "books": set()})
    for r in rows:
        c = counts[f"{r['split']}|{r['stratum']}|{r['family']}"]
        c["clips"] += 1
        c["speakers_or_voices"].add(r["voice"] or r["speaker"])
        c["books"].add(r["book"])
    report = {
        "inputs": {
            "mlaad_revision": MLAAD_REVISION,
            "mlaad_meta_sha256": {f: file_sha(root / MLAAD_DIR / f / "meta.csv") for f in FAMILIES},
            "teacher_eval_clips_sha256": file_sha(root / TEACHER_CSV),
            "rvcbench_pairs_sha256": file_sha(pairs_path),
            "vcc2018_zip_sha256": {z: file_sha(root / "development/vcc2018" / z) for z in VCC_ZIPS},
        },
        "manifest_sha256": {s: file_sha(out / f"manifest-{s}.csv") for s in ("calibration", "evaluation")},
        "counts_before_4s_eligibility": {k: {"clips": v["clips"], "speakers_or_voices": len(v["speakers_or_voices"]),
                                             "books": len(v["books"] - {""})} for k, v in sorted(counts.items())},
        "dropped": dict(sorted(dropped.items())),
        "checks": checks,
        "generator_versions": {
            f: f"MLAAD @{MLAAD_REVISION[:7]} model_name={f!r}; checkpoint/API version not stated by the publisher"
            for f in FAMILIES
        } | {"proxy/" + k: v for k, v in PROXY_MODELS.items()},
        "notes": [
            "Development = teacher_eval MLAAD sample (250/family, 3 M-AILABS US readers); excluded with its texts.",
            "Chatterbox and FishTTS rows state no reference voice; voice disjointness is unverifiable for them.",
            "ElevenLabs/Gemini stock voices recur in development, calibration fakes and evaluation; the cutoff "
            "uses calibration genuine only, so no fake voice informs it.",
            "LIMITATION: evaluation fakes' matched genuine parents (Elizabeth Klett, M-AILABS en_UK.tgz) are NOT on "
            "Drive and are not matched; evaluation genuine comes from other sources (LibriTTS test-clean via "
            "RVCBench; VCC 2018) than the fakes (M-AILABS texts), so false alarms and detection are measured on "
            "different recording sources.",
            "RVCBench is CC BY-NC 4.0 + per-system terms; MLAAD CC BY-NC 4.0: research only, never app assets.",
        ],
    }
    (out / "build.json").write_text(json.dumps(report, indent=1))
    return report


class IntakeReader:
    """Read-only, checksum-verified reader of the modern-intake clip cache (treated 4 s crops)."""

    def __init__(self, root):
        from .clip_cache import fingerprint

        base = root / INTAKE / "clip-cache/fixed4s-rmsnorm-v1"
        if json.loads((base / "meta.json").read_text())["fingerprint"] != fingerprint():
            raise ValueError("Intake cache preprocessing differs from clip_cache.treat")
        self.base, self.entries = base, {}
        for p in sorted((base / "manifest").glob("*.jsonl")):
            for line in p.read_text().splitlines():
                e = json.loads(line)
                self.entries[(e["source"], e["source_id"])] = e

    def get(self, source, sid):
        e = self.entries[(source, sid)]
        if not e["eligible"]:
            return None
        with (self.base / "shards" / e["shard"]).open("rb") as f:
            f.seek(e["offset"])
            b = f.read(e["size"])
        if hashlib.sha256(b).hexdigest() != e["sha256"]:
            raise ValueError(f"Intake checksum mismatch {sid}")
        return b


def score(root, out, provenance, encoder, splits):
    import numpy as np
    import torch
    from scipy.special import expit

    from detector.audio import decode as det_decode
    from detector.audio import pcm_hash
    from detector.inference import Detector

    from .audio import channel, decode, model_view
    from .clip_cache import treat
    from .data import digest
    from .models import Teacher
    from .teacher_eval import mean_logit

    from detector.model.ssl import ENCODERS

    enc = Path(ENCODERS["wav2vec2-xls-r-300m"])
    for name, want in ENCODER_SHA256.items():
        if file_sha(enc / name) != want:
            raise ValueError(f"Release encoder differs from the training copy: {name}")
    det = Detector()
    prov = json.loads(Path(provenance).read_text())
    if prov["provenance"]["audit_manifest_sha256"] != det.release["audit_manifest_sha256"]:
        raise ValueError("Provenance is not from the release's audit manifest")
    trained = set(prov["hashes"])
    runs = root / TEACHER_RUNS
    frozen = json.loads((runs / "teacher-frozen.json").read_text())
    if digest(runs / "teacher.best.pt") != frozen["teacher_sha256"]:
        raise ValueError("Teacher differs from its frozen record")
    (pa, pb) = frozen["platt"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(runs / "teacher.best.pt", map_location="cpu", weights_only=True)
    teacher = Teacher(encoder).to(device)
    teacher.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    teacher.eval()
    binding = {
        "release_sha256": det.release_sha256,
        "release_members": [m["checkpoint_sha256"] for m in det.release["members"]],
        "release_encoder_sha256": ENCODER_SHA256,
        "teacher_sha256": frozen["teacher_sha256"], "teacher_platt": [pa, pb],
        "manifest_sha256": {s: file_sha(out / f"manifest-{s}.csv") for s in splits},
        "provenance_audit_manifest_sha256": prov["provenance"]["audit_manifest_sha256"],
        "view": "clip_cache.treat 4 s crop + norm, clean channel (release 'paired' view; teacher Platt p)",
    }
    bpath = out / f"score-inputs-{'-'.join(splits)}.json"
    if bpath.exists() and json.loads(bpath.read_text()) != binding:
        raise ValueError("Scoring inputs changed")
    bpath.write_text(json.dumps(binding, indent=1))
    sdir = out / "scores"
    sdir.mkdir(exist_ok=True)
    intake = None
    zips = {}
    rows = []
    for s in splits:
        with (out / f"manifest-{s}.csv").open() as f:
            rows += [r for r in csv.DictReader(f) if r["source"] != "libri-dev"]
    todo = [r for r in rows if not (sdir / (hashlib.sha256(r["id"].encode()).hexdigest() + ".json")).exists()]
    print(f"score: {len(rows)} rows, {len(todo)} pending", flush=True)
    with torch.inference_mode():
        # Reproduction check on 5 development clips: same models + view must give recorded scores.
        with (root / TEACHER_CSV).open() as f:
            ref = [r for r in csv.DictReader(f) if r["panel"] == "mlaad" and r["channel"] == "clean"
                   and r["label"] == "1"][:5]
        repro = []
        for r in ref:
            crop = treat((root / "development" / r["id"].removeprefix("mlaad/")).read_bytes())
            x = channel(decode(crop), "clean")
            pr = det.analyze_wave(x.astype(np.float32))["fake_probability"]
            pt = float(expit(pa * mean_logit(teacher, model_view(x, {"pair": "phonemedf"}), device) + pb))
            fb = json.loads((root / "v2/day-target-1/fallback-detector-v1/scores" /
                             (hashlib.sha256(r["id"].encode()).hexdigest() + ".json")).read_text())
            repro.append({"id": r["id"], "teacher": [pt, float(r["p"])],
                          "release": [pr, fb["clean"]["paired"]["p"]]})
        print("repro", json.dumps(repro), flush=True)
        if any(abs(v["teacher"][0] - v["teacher"][1]) > 2e-3 or abs(v["release"][0] - v["release"][1]) > 2e-3
               for v in repro):
            raise ValueError("Scores do not reproduce the recorded development scores")
        (out / f"repro-{'-'.join(splits)}.json").write_text(json.dumps(repro, indent=1))
        for i, r in enumerate(todo, 1):
            raw, overlap = None, None
            if r["source"] == "mlaad":
                raw = (root / r["locator"]).read_bytes()
            elif r["source"] == "vcc2018":
                z, m = r["locator"].split("::")
                zf = zips.get(z) or zips.setdefault(z, zipfile.ZipFile(root / "development/vcc2018" / z))
                raw = zf.read(m)
            if raw is not None:
                crop = treat(raw)
                overlap = pcm_hash(det_decode(raw)[0]) in trained
            else:  # rvcbench: cached treated crop; raw PCM not cached here
                intake = intake or IntakeReader(root)
                crop = intake.get("rvcbench", r["locator"])
            rec = {"id": r["id"], "eligible": crop is not None, "exact_release_train_dev_overlap": overlap}
            if crop is not None:
                rec["crop_sha256"] = hashlib.sha256(crop).hexdigest()
                x = channel(decode(crop), "clean")
                rec["p_release"] = det.analyze_wave(x.astype(np.float32))["fake_probability"]
                s = mean_logit(teacher, model_view(x, {"pair": "phonemedf"}), device)
                rec["p_teacher"] = None if s is None else float(expit(pa * s + pb))
            tmp = sdir / (hashlib.sha256(r["id"].encode()).hexdigest() + ".tmp")
            tmp.write_text(json.dumps(rec))
            tmp.rename(tmp.with_suffix(".json"))
            if i % 100 == 0:
                print(f"score {i}/{len(todo)}", flush=True)
    return binding


EPS = 1e-6
# Release v1 encoder files (detector/pretrained/wav2vec2-xls-r-300m), as verified by day_fallback.
ENCODER_SHA256 = {
    "config.json": "0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac",
    "preprocessor_config.json": "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335",
    "pytorch_model.bin": "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0",
}


def _z(p, mu, sd):
    import math

    p = min(max(p, EPS), 1 - EPS)
    return (math.log(p / (1 - p)) - mu) / sd


def report(root, out, candidate_path, libri_release_path):
    cand = json.loads(Path(candidate_path).read_text())
    libri = json.loads(Path(libri_release_path).read_text())
    if libri["release_sha256"] != cand["release_sha256"]:
        raise ValueError("libri-dev release scores come from a different release")
    libri = libri["scores"]
    with (root / TEACHER_CSV).open() as f:
        teacher_libri = {r["id"]: float(r["p"]) for r in csv.DictReader(f)
                         if r["channel"] == "clean" and r["panel"].startswith("libri-dev")}
    rules = candidate_rules(cand)
    if cand["cutoff"] != cand["cutoff"] or isinstance(cand["cutoff"], str):
        raise ValueError("Candidate cutoff is not frozen")
    res = {"candidate_file_sha256": file_sha(candidate_path), "rules": {k: v["describe"] for k, v in rules.items()}}
    for split in ("calibration", "evaluation"):
        with (out / f"manifest-{split}.csv").open() as f:
            rows = list(csv.DictReader(f))
        scored, inel = [], Counter()
        for r in rows:
            if r["source"] == "libri-dev":
                pr = libri[r["id"]]
                pr = pr["p"] if isinstance(pr, dict) else pr
                scored.append(dict(r, p_release=pr, p_teacher=teacher_libri[r["id"]]))
                continue
            rec = json.loads((out / "scores" / (hashlib.sha256(r["id"].encode()).hexdigest() + ".json")).read_text())
            if not rec["eligible"] or rec.get("p_release") is None or rec.get("p_teacher") is None:
                inel[f"{r['stratum']}|{r['family']}"] += 1
                continue
            if rec.get("exact_release_train_dev_overlap"):
                inel[f"{r['stratum']}|{r['family']}|exact_train_overlap"] += 1
                continue
            scored.append(dict(r, p_release=rec["p_release"], p_teacher=rec["p_teacher"]))
        res[split] = {"ineligible_shorter_than_4s_or_overlap": dict(inel)}
        for name, rule in rules.items():
            cells = defaultdict(lambda: {"flagged": 0, "n": 0, "speakers": set()})
            for r in scored:
                c = cells[f"{r['stratum']}|{r['family']}"]
                c["n"] += 1
                c["flagged"] += rule["flag"](r)
                c["speakers"].add(r["voice"] or r["speaker"])
                if r["label"] == "0":
                    s = cells[f"_genuine_by_speaker|{r['stratum']}|{r['speaker']}"]
                    s["n"] += 1
                    s["flagged"] += rule["flag"](r)
                    s["speakers"].add(r["speaker"])
            res[split][name] = {k: {"flagged": v["flagged"], "n": v["n"], "rate": round(v["flagged"] / v["n"], 4),
                                    "speakers_or_voices": len(v["speakers"])} for k, v in sorted(cells.items())}
            if split == "evaluation":
                fam = {f: cells[f"mlaad|{f}"] for f in FAMILIES}
                g = cells["libritts-test-clean|genuine"]
                res[split][name]["_target_primary"] = {
                    "genuine_fa": [g["flagged"], g["n"]],
                    "families": {f: [v["flagged"], v["n"]] for f, v in fam.items()},
                    "met": g["n"] > 0 and g["flagged"] <= 0.15 * g["n"]
                    and all(v["n"] and v["flagged"] >= 0.70 * v["n"] for v in fam.values()),
                }
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


def candidate_rules(cand):
    """Rules from the externally frozen candidate file (modern-tts-70-at-15)."""
    n = cand["normaliser"] if "normaliser" in cand else cand
    mr, sr = n["release"]["mu"], n["release"]["sd"]
    mt, st = n["teacher"]["mu"], n["teacher"]["sd"]
    w = cand.get("weight_release", 0.5)
    c = cand["cutoff"]
    singles = {m: v["cutoff_p"] for m, v in cand.get("single_model_baselines_same_rule", {}).items()}
    rules = {"combo": {"describe": f"{w}*z_release+{1 - w}*z_teacher >= {c}",
                       "flag": lambda r: w * _z(r["p_release"], mr, sr) + (1 - w) * _z(r["p_teacher"], mt, st) >= c}}
    if "release" in singles:
        rules["release_only"] = {"describe": f"p_release >= {singles['release']}",
                                 "flag": lambda r: r["p_release"] >= singles["release"]}
    if "teacher" in singles:
        rules["teacher_only"] = {"describe": f"p_teacher >= {singles['teacher']}",
                                 "flag": lambda r: r["p_teacher"] >= singles["teacher"]}
    return rules


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["build", "score", "report"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--pairs", type=Path, help="modern-intake rvcbench/pairs.jsonl")
    p.add_argument("--provenance", type=Path)
    p.add_argument("--encoder", type=Path, help="teacher XLS-R directory")
    p.add_argument("--splits", nargs="+", default=["evaluation"])
    p.add_argument("--candidate", type=Path)
    p.add_argument("--libri-release", type=Path)
    a = p.parse_args()
    out = a.out or a.data / "v2/independent-panel-1"
    if a.phase == "build":
        print(json.dumps(build(a.data, out, a.pairs), indent=1))
    elif a.phase == "score":
        print(json.dumps(score(a.data, out, a.provenance, a.encoder, a.splits), indent=1))
    else:
        print(json.dumps(report(a.data, out, a.candidate, a.libri_release), indent=1))


if __name__ == "__main__":
    main()
