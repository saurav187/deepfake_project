"""V2 overnight independent evaluation: sign-off, secondary channels and the single report.

Protocol: college/27_V2_INDEPENDENT_EVAL_PROTOCOL.md (committed before any evaluation score).
Panel and clean scores: v2/independent-panel-1 (independent_panel.py, owned separately; read-only).
Frozen candidate: v2/modern-tts-70-at-15/frozen-candidate-cutoff.json (read-only).
Outputs: <data>/v2/overnight-v2/ only.

  signoff   bind protocol, manifests, candidate and code by sha256 with a timestamp; refuses if any
            evaluation-split score already exists.
  channels  opus + simulated phone scores (release v1 + Teacher V1) of evaluation rows, after sign-off
  report    one pass at the frozen cutoff; refuses if any bound hash changed
"""

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

FAMILIES = ["Chatterbox", "ElevenLabs-v3", "FishTTS", "Gemini-3.1-Flash-TTS"]
PANEL = "v2/independent-panel-1"
CAND = "v2/modern-tts-70-at-15/frozen-candidate-cutoff.json"
CAND_SHA = "8c237786595fcded000d5cc1fca898584105583dc5ab43c9b4ce5e5db430a22e"
OUT = "v2/overnight-v2"
SHIPPED_LIKELY_FAKE = 0.5623497986447662  # detector/runs/release/v1/release.json thresholds
RELEASE_SHA = "bdb7748cd609bc7a30b86ab83dceb3c3b53c80d710826fa8c6d50d2a3ecab4fc"
PRIMARY_GENUINE = "libritts-test-clean"
TARGET_DET, FA_LIMIT = 0.70, 0.15
MIN_FAKES, MIN_GEN_SPK, MIN_GEN_REC = 30, 10, 100
BOOT, SEED = 2000, 20261126
EPS = 1e-6
CHANNELS = ("opus", "phone")
IST = timezone(timedelta(hours=5, minutes=30))


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def key(uid):
    return hashlib.sha256(uid.encode()).hexdigest()


def now():
    t = datetime.now(timezone.utc)
    return {"utc": t.isoformat(timespec="seconds"), "ist": t.astimezone(IST).isoformat(timespec="seconds")}


def manifest(root, split):
    with (root / PANEL / f"manifest-{split}.csv").open() as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ sign-off
def signoff(root, protocol, commit):
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol-signoff.json"
    if path.exists():
        raise SystemExit(f"{path} exists; sign-off is write-once")
    cand = root / CAND
    if sha(cand) != CAND_SHA:
        raise ValueError("Frozen candidate cutoff file differs from the protocol")
    ev = manifest(root, "evaluation")
    present = [r["id"] for r in ev if (root / PANEL / "scores" / (key(r["id"]) + ".json")).exists()]
    present += [r["id"] for r in ev if (out / "scores-channels" / (key(r["id"]) + ".json")).exists()]
    if present:
        raise ValueError(f"{len(present)} evaluation scores already exist; the test is not untouched")
    c = json.loads(cand.read_text())
    rec = {
        "kind": "V2 independent evaluation sign-off (write-once)",
        "signed": now(),
        "signed_by": "v2-overnight-lead (owner assignment college/26_V2_OVERNIGHT_ASSIGNMENT.md)",
        "protocol_sha256": sha(protocol),
        "protocol_path": "college/27_V2_INDEPENDENT_EVAL_PROTOCOL.md",
        "code_commit": commit,
        "report_code_sha256": sha(__file__),
        "candidate_cutoff_file_sha256": CAND_SHA,
        "primary": {"rule": "0.5*z_release + 0.5*z_teacher >= cutoff", "cutoff": c["cutoff"],
                    "normaliser": c["normaliser"], "release_sha256": c["release_sha256"],
                    "teacher_sha256": c["teacher_sha256"]},
        "references": {
            "R1_release_calibrated": c["single_model_baselines_same_rule"]["release"]["cutoff_p"],
            "R2_release_shipped": SHIPPED_LIKELY_FAKE,
            "T_teacher_calibrated": c["single_model_baselines_same_rule"]["teacher"]["cutoff_p"],
        },
        "panel_build_sha256": sha(root / PANEL / "build.json"),
        "manifest_sha256": {s: sha(root / PANEL / f"manifest-{s}.csv") for s in ("calibration", "evaluation")},
        "evaluation_rows": dict(Counter(f"{r['stratum']}|{r['family']}" for r in ev)),
        "evaluation_scores_present_at_signoff": 0,
        "decision_rule": {"genuine_stratum": PRIMARY_GENUINE, "fa_limit": FA_LIMIT, "det_target": TARGET_DET,
                          "families": FAMILIES, "insufficient_if": {"fakes_per_family_lt": MIN_FAKES,
                                                                    "genuine_speakers_lt": MIN_GEN_SPK,
                                                                    "genuine_recordings_lt": MIN_GEN_REC}},
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n")
    tmp.rename(path)
    return rec | {"signoff_sha256": sha(path)}


def check_signoff(root):
    s = json.loads((root / OUT / "protocol-signoff.json").read_text())
    if sha(root / CAND) != s["candidate_cutoff_file_sha256"]:
        raise ValueError("Candidate changed after sign-off")
    for k, v in s["manifest_sha256"].items():
        if sha(root / PANEL / f"manifest-{k}.csv") != v:
            raise ValueError(f"manifest-{k}.csv changed after sign-off")
    return s


# ------------------------------------------------------------------ channels
def channels(root, encoder, limit=None):
    """Opus/phone scores of evaluation rows with the frozen release and teacher (same crop view)."""
    import numpy as np
    import torch
    from scipy.special import expit

    from detector.inference import Detector

    from .audio import channel, decode, model_view
    from .data import digest
    from .independent_panel import IntakeReader, TEACHER_RUNS, VCC_ZIPS
    from .clip_cache import treat
    from .models import Teacher
    from .teacher_eval import mean_logit

    check_signoff(root)
    det = Detector()
    if det.release_sha256 != RELEASE_SHA:
        raise ValueError("Release differs from the frozen candidate")
    runs = root / TEACHER_RUNS
    frozen = json.loads((runs / "teacher-frozen.json").read_text())
    if digest(runs / "teacher.best.pt") != frozen["teacher_sha256"]:
        raise ValueError("Teacher differs from its frozen record")
    pa, pb = frozen["platt"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(runs / "teacher.best.pt", map_location="cpu", weights_only=True)
    teacher = Teacher(encoder).to(device)
    teacher.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    teacher.eval()
    sdir = root / OUT / "scores-channels"
    sdir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in manifest(root, "evaluation")]
    todo = [r for r in rows if not (sdir / (key(r["id"]) + ".json")).exists()][:limit]
    print(f"channels: {len(rows)} rows, {len(todo)} pending", flush=True)
    import zipfile
    zips, intake = {}, None
    t0 = time.time()
    with torch.inference_mode():
        for i, r in enumerate(todo, 1):
            if r["source"] == "mlaad":
                crop = treat((root / r["locator"]).read_bytes())
            elif r["source"] == "vcc2018":
                z, m = r["locator"].split("::")
                zf = zips.get(z) or zips.setdefault(z, zipfile.ZipFile(root / "development/vcc2018" / z))
                crop = treat(zf.read(m))
            else:
                intake = intake or IntakeReader(root)
                crop = intake.get("rvcbench", r["locator"])
            rec = {"id": r["id"], "eligible": crop is not None}
            if crop is not None:
                rec["crop_sha256"] = hashlib.sha256(crop).hexdigest()
                base = decode(crop)
                for ch in CHANNELS:
                    x = channel(base, ch).astype(np.float32)
                    s = mean_logit(teacher, model_view(x, {"pair": "phonemedf"}), device)
                    rec[ch] = {"p_release": det.analyze_wave(x)["fake_probability"],
                               "p_teacher": None if s is None else float(expit(pa * s + pb))}
            tmp = sdir / (key(r["id"]) + ".tmp")
            tmp.write_text(json.dumps(rec))
            tmp.rename(tmp.with_suffix(".json"))
            if i % 100 == 0:
                print(f"channels {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    return {"scored": len(todo)}


# ------------------------------------------------------------------ report
def z(p, mu, sd):
    p = min(max(p, EPS), 1 - EPS)
    return (math.log(p / (1 - p)) - mu) / sd


def rules(s):
    pr, n, c = s["primary"], s["primary"]["normaliser"], s["primary"]["cutoff"]
    ref = s["references"]
    combo = lambda r: 0.5 * z(r["p_release"], n["release"]["mu"], n["release"]["sd"]) \
        + 0.5 * z(r["p_teacher"], n["teacher"]["mu"], n["teacher"]["sd"]) >= c
    return {
        "PRIMARY_combo": (combo, f"0.5*z_release+0.5*z_teacher >= {c}"),
        "R1_release_calibrated": (lambda r: r["p_release"] >= ref["R1_release_calibrated"],
                                  f"p_release >= {ref['R1_release_calibrated']}"),
        "R2_release_shipped": (lambda r: r["p_release"] >= ref["R2_release_shipped"],
                               f"p_release >= {ref['R2_release_shipped']} (single 4 s crop)"),
        "T_teacher_calibrated": (lambda r: r["p_teacher"] >= ref["T_teacher_calibrated"],
                                 f"p_teacher >= {ref['T_teacher_calibrated']}"),
    }


def wilson(k, n, zc=1.959964):
    if not n:
        return None
    p = k / n
    d = 1 + zc * zc / n
    m = (p + zc * zc / (2 * n)) / d
    h = zc * math.sqrt(p * (1 - p) / n + zc * zc / (4 * n * n)) / d
    return [round(m - h, 4), round(m + h, 4)]


def cluster_boot(flags_by_cluster):
    """95% percentile interval of the pooled rate, resampling clusters with replacement."""
    cl = list(flags_by_cluster.values())
    if len(cl) < 2:
        return None
    rng = random.Random(SEED)
    rates = []
    for _ in range(BOOT):
        k = n = 0
        for _ in cl:
            c = cl[rng.randrange(len(cl))]
            k += sum(c)
            n += len(c)
        rates.append(k / n)
    rates.sort()
    return [round(rates[int(0.025 * BOOT)], 4), round(rates[int(0.975 * BOOT) - 1], 4)]


def select_rows(root, split):
    """One row per original per condition; identical row set for every rule."""
    rows, excl = [], Counter()
    seen = {}
    for r in sorted(manifest(root, split), key=lambda r: key(r["id"])):
        unit = (r["stratum"], r["family"], r["parent"])
        if unit in seen:
            excl[f"{r['stratum']}|{r['family']}|duplicate_parent"] += 1
            continue
        seen[unit] = r["id"]
        rows.append(r)
    return rows, excl


def scores_for(root, r, ch):
    if r["source"] == "libri-dev":
        return None
    if ch == "clean":
        p = root / PANEL / "scores" / (key(r["id"]) + ".json")
        if not p.exists():
            return "missing"
        rec = json.loads(p.read_text())
        if not rec.get("eligible"):
            return "ineligible"
        if rec.get("exact_release_train_dev_overlap"):
            return "exact_train_overlap"
        if rec.get("p_release") is None or rec.get("p_teacher") is None:
            return "ineligible"
        return {"p_release": rec["p_release"], "p_teacher": rec["p_teacher"],
                "overlap_checked": rec.get("exact_release_train_dev_overlap") is not None}
    p = root / OUT / "scores-channels" / (key(r["id"]) + ".json")
    if not p.exists():
        return "missing"
    rec = json.loads(p.read_text())
    if not rec.get("eligible") or rec.get(ch, {}).get("p_teacher") is None:
        return "ineligible"
    return rec[ch]


def tabulate(rows, rule):
    cells = defaultdict(lambda: {"flags": [], "clusters": defaultdict(list), "speakers": set()})
    for r in rows:
        f = int(bool(rule(r)))
        c = cells[f"{r['stratum']}|{r['family']}"]
        c["flags"].append(f)
        cl = r["speaker"] if r["label"] == "0" else (r["voice"] if not r["voice"].endswith(":unstated") else None)
        if cl:
            c["clusters"][cl].append(f)
        c["speakers"].add(r["voice"] or r["speaker"])
    out = {}
    for k, c in sorted(cells.items()):
        n, kf = len(c["flags"]), sum(c["flags"])
        unstated = len(c["clusters"]) == 0
        out[k] = {"flagged": kf, "n": n, "rate": round(kf / n, 4), "wilson95": wilson(kf, n),
                  "clusters": len(c["clusters"]) if not unstated else "unknown",
                  "cluster_boot95": None if unstated else cluster_boot(c["clusters"])}
    return out


def decide(t, gen_rows):
    gk = f"{PRIMARY_GENUINE}|genuine"
    g = t.get(gk)
    fam = {f: t.get(f"mlaad|{f}") for f in FAMILIES}
    gaps, untested, small = {}, [f for f, v in fam.items() if not v], []
    if not g:
        return {"verdict": "INSUFFICIENT INDEPENDENT EVIDENCE", "reason": "no primary genuine scored"}
    allowed = math.floor(FA_LIMIT * g["n"])
    if g["flagged"] > allowed:
        gaps["genuine_FA"] = {"flagged": g["flagged"], "allowed": allowed, "excess_recordings": g["flagged"] - allowed}
    for f, v in fam.items():
        if not v:
            continue
        need = math.ceil(TARGET_DET * v["n"] - 1e-9)
        if v["flagged"] < need:
            gaps[f] = {"detected": v["flagged"], "n": v["n"], "needed": need, "short_by": need - v["flagged"]}
        if v["n"] < MIN_FAKES:
            small.append(f)
    spk = len({r["speaker"] for r in gen_rows})
    if spk < MIN_GEN_SPK or g["n"] < MIN_GEN_REC:
        small.append("genuine")
    robust = all(
        (v["cluster_boot95"] or v["wilson95"])[0] >= TARGET_DET for v in fam.values() if v
    ) and (g["cluster_boot95"] or g["wilson95"])[1] <= FA_LIMIT
    if untested or small:
        verdict = "INSUFFICIENT INDEPENDENT EVIDENCE"
    elif gaps:
        verdict = "MISS"
    else:
        verdict = "PASS" if robust else "PASS at point estimate; not robust"
    return {"verdict": verdict, "gaps": gaps, "untested_families": untested, "too_small": small,
            "genuine_speakers": spk}


def report(root):
    s = check_signoff(root)
    rset = rules(s)
    res = {"signoff_sha256": sha(root / OUT / "protocol-signoff.json"), "reported": now(),
           "rules": {k: v[1] for k, v in rset.items()}}
    for split in ("calibration", "evaluation"):
        rows, excl = select_rows(root, split)
        res[split] = {"excluded_units": dict(excl)}
        for ch in ("clean",) + CHANNELS:
            if split == "calibration" and ch != "clean":
                continue
            scored, inel, unchecked = [], Counter(), Counter()
            for r in rows:
                sc = scores_for(root, r, ch)
                if sc is None:
                    continue  # libri-dev: cutoff source, reported in the freeze record
                if isinstance(sc, str):
                    inel[f"{r['stratum']}|{r['family']}|{sc}"] += 1
                    continue
                if ch == "clean" and not sc.get("overlap_checked", True):
                    unchecked[f"{r['stratum']}|{r['family']}"] += 1
                scored.append(dict(r, **sc))
            block = {"excluded": dict(inel), "exact_overlap_unchecked": dict(unchecked)}
            for name, (rule, _) in rset.items():
                t = tabulate(scored, rule)
                block[name] = {"cells": t}
                if split == "evaluation":
                    gen = [r for r in scored if r["stratum"] == PRIMARY_GENUINE]
                    block[name]["decision"] = decide(t, gen)
                    per = defaultdict(lambda: [0, 0])
                    for r in gen:
                        per[r["speaker"]][0] += int(bool(rule(r)))
                        per[r["speaker"]][1] += 1
                    block[name]["genuine_fa_by_speaker"] = dict(sorted(per.items()))
            res[split][ch] = block
    out = root / OUT
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    (out / "result.md").write_text(render(res))
    return res


def render(res):
    lines = [f"# V2 independent evaluation result ({res['reported']['ist']} IST)", ""]
    for ch in ("clean",) + CHANNELS:
        b = res["evaluation"].get(ch)
        if not b:
            continue
        lines += [f"## Evaluation, {ch}" + (" (primary)" if ch == "clean" else " (secondary)"), "",
                  "| Rule | Genuine FA (LibriTTS test-clean) | " + " | ".join(FAMILIES) + " | Verdict |",
                  "|---|---|" + "---|" * len(FAMILIES) + "---|"]
        for name, v in b.items():
            if not isinstance(v, dict) or "cells" not in v:
                continue
            c = v["cells"]
            cell = lambda k: (f"{c[k]['flagged']}/{c[k]['n']} ({100 * c[k]['rate']:.1f}%)" if k in c else "untested")
            lines.append(f"| {name} | {cell(PRIMARY_GENUINE + '|genuine')} | "
                         + " | ".join(cell('mlaad|' + f) for f in FAMILIES) + f" | {v['decision']['verdict']} |")
        lines += ["", f"Excluded: {b['excluded']}", ""]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["signoff", "channels", "report"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--protocol", type=Path)
    p.add_argument("--commit")
    p.add_argument("--encoder", type=Path)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    if a.phase == "signoff":
        print(json.dumps(signoff(a.data, a.protocol, a.commit), indent=1))
    elif a.phase == "channels":
        print(json.dumps(channels(a.data, a.encoder, a.limit)))
    else:
        r = report(a.data)
        print(json.dumps({k: r["evaluation"][k][n]["decision"] for k in ("clean",) + CHANNELS
                          if k in r["evaluation"] for n in r["evaluation"][k] if n.startswith("PRIMARY")}, indent=1))


if __name__ == "__main__":
    main()
