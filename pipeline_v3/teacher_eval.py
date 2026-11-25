"""Freeze Teacher V1 and its threshold, then evaluate on non-sealed development data.

  freeze    teacher.best.pt sha256 + Platt calibration (calibrate fold) + thresholds
            (threshold fold, the release rule in export.py) -> runs/teacher-frozen.json.
            Written once, before any validate or panel scoring.
  evaluate  scores ONLY with the frozen record: validate fold under fixed channels, plus
            optional development panels (LibriSpeech dev genuine, MLAAD matched panel),
            preprocessed exactly like training (4 s fixed crop -> frozen norm).
            -> runs/teacher-eval.json with clip, reader and voice counts per cell.

Verdict: p = sigmoid(a * mean_window_logit + b); likely_fake if p >= hi, likely_real if
p < lo, otherwise uncertain. False alarm = genuine clip judged likely_fake.

  python -m pipeline_v3.teacher_eval freeze   --data $W --local $L --encoder $D/models/xlsr
  python -m pipeline_v3.teacher_eval evaluate --data $W --local $L --encoder $D/models/xlsr \
      --dev $D [--panels libri-dev mlaad]
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CHANNELS = ("clean", "opus", "mp3", "aac", "phone", "rate", "playback", "noise")
PANEL_CHANNELS = ("clean", "opus", "phone")
LIBRI_PER_READER = 20
MLAAD_PER_FAMILY = 250
UNMEASURED = {
    "hindi": "unmeasured: Hindi is blocked (select readers < 5); no Hindi rows in this pack",
    "genuine_phone_recordings": "unmeasured: no licensed genuine phone/WhatsApp recordings "
    "on Drive; 'phone'/'opus' channels here are simulations of clean recordings",
    "melotts": "unmeasured: MeloTTS excluded from this experiment",
    "sealed_sets": "reserved for the final comparison; not evaluated",
    "student": "deferred",
}


def _model(encoder, device):

    from .models import Teacher

    return Teacher(encoder).to(device)


def mean_logit(model, x, device):
    from .audio import windows
    from .train import logits

    if not list(windows(x)):
        return None
    return float(logits(model, x, False, device).mean().cpu())


def teacher_context(cfg):
    from .cli import bound_audit

    audit = bound_audit(cfg.data, cfg.local / "index.json")
    pilot = json.loads((cfg.data / "pilot.json").read_text())
    from .exploratory import pilot_authorizes

    if not pilot_authorizes(cfg.data, pilot, audit):
        raise ValueError("Passing pilot (or a recorded exploratory advisory) must match this pack, language and exception")
    return audit


def load_teacher(cfg, best, device, audit):
    import torch

    state = torch.load(best, map_location="cpu", weights_only=True)
    if audit.get("exception"):
        if (state.get("index_sha256") != audit["index_sha256"]
                or state.get("exception") != audit["exception"]
                or state.get("encoder") != str(cfg.encoder)):
            raise ValueError("Teacher checkpoint differs from this pack or exception")
        state = state["model"]
    model = _model(cfg.encoder, device)
    model.load_state_dict(state)
    return model.eval()


def freeze(cfg):
    import torch
    from scipy.special import expit

    from .audio import decode, model_view
    from .data import digest, entries, payload
    from .export import platt

    runs = cfg.data / "runs"
    out = runs / "teacher-frozen.json"
    if out.exists():
        raise ValueError("Teacher and threshold are already frozen")
    audit = teacher_context(cfg)
    best = runs / "teacher.best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_teacher(cfg, best, device, audit)

    def collect(split):
        out_rows = []
        with torch.inference_mode():
            for r in entries(cfg.local, {split}):
                s = mean_logit(model, model_view(decode(payload(cfg.local, r)), r), device)
                if s is not None:
                    out_rows.append((r, s))
        return out_rows

    cal = collect("calibrate")
    a, b = platt([s for _, s in cal], [int(r["label"]) for r, _ in cal])
    thr = collect("threshold")
    groups = [[float(expit(a * s + b)) for r, s in thr if r["label"] == lab] for lab in "01"]
    if min(map(len, groups)) < 50:
        raise ValueError("Need at least 50 threshold-fold recordings per class")
    lo = min(0.40, float(np.quantile(groups[1], 0.02, method="lower")))
    hi = max(0.60, float(np.nextafter(np.quantile(groups[0], 0.98, method="higher"), 1.0)))
    record = {
        "teacher_sha256": digest(best),
        "index_sha256": digest(cfg.local / "index.json"),
        "platt": [a, b],
        "thresholds": [lo, hi],
        "rule": "export.py release rule on the teacher: lo = min(0.40, q02 fake), "
        "hi = max(0.60, next above q98 genuine), threshold fold, clean audio",
        "calibrate_rows": len(cal),
        "threshold_rows": {"genuine": len(groups[0]), "fake": len(groups[1])},
        "threshold_fold_false_alarm": float(np.mean(np.array(groups[0]) >= hi)),
        "threshold_fold_detection": float(np.mean(np.array(groups[1]) >= hi)),
        "exception": audit.get("exception"),
        "pilot_go": json.loads((cfg.data / "pilot.json").read_text()).get("go"),
        "pilot_advisory": json.loads((cfg.data / "pilot-advisory.json").read_text())
        if (cfg.data / "pilot-advisory.json").exists() else None,
    }
    runs.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(out)
    return record


def verdict(p, lo, hi):
    return "likely_fake" if p >= hi else "likely_real" if p < lo else "uncertain"


def summarise(items, key):
    """items: dicts with label, generator, speaker, voice, channel, verdict, p."""
    cells = defaultdict(list)
    for it in items:
        cells[key(it)].append(it)
    out = {}
    for k, v in sorted(cells.items()):
        n = len(v)
        fake = v[0]["label"] == "1"
        out["/".join(k)] = {
            "clips": n,
            "readers": len({x["speaker"] for x in v}),
            "voices": len({x["voice"] for x in v if x.get("voice") not in {None, "", "-", "unknown"}}) if fake else 0,
            "likely_fake": sum(x["verdict"] == "likely_fake" for x in v),
            "uncertain": sum(x["verdict"] == "uncertain" for x in v),
            "likely_real": sum(x["verdict"] == "likely_real" for x in v),
            ("detection_rate" if fake else "false_alarm_rate"): float(
                np.mean([x["verdict"] == "likely_fake" for x in v])
            ),
        }
    return out


def failures(items, label):
    """Genuine (label 0): false alarms per source x channel. Fake (label 1): misses per
    source x generator x channel; 'missed' = likely_real, 'uncertain' reported separately."""
    out = {}
    for it in items:
        if it["label"] != label:
            continue
        key = (it["panel"], it["channel"]) if label == "0" else (it["panel"], it["generator"], it["channel"])
        c = out.setdefault("/".join(key), {"clips": 0, "readers": set(), "likely_fake": 0,
                                            "uncertain": 0, "likely_real": 0})
        c["clips"] += 1
        c["readers"].add(it["speaker"])
        c[it["verdict"]] += 1
    for c in out.values():
        c["readers"] = len(c["readers"])
        if label == "0":
            c["false_alarm_rate"] = c["likely_fake"] / c["clips"]
        else:
            c["missed_rate"] = c["likely_real"] / c["clips"]
            c["not_detected_rate"] = (c["likely_real"] + c["uncertain"]) / c["clips"]
    return dict(sorted(out.items()))


def separability(items):
    from .metrics import metrics

    out = {}
    for ch in sorted({x["channel"] for x in items}):
        gen = [x for x in items if x["channel"] == ch and x["label"] == "0"]
        for g in sorted({x["generator"] for x in items if x["label"] == "1"}):
            fk = [x for x in items if x["channel"] == ch and x["generator"] == g]
            parents = {x["parent"] for x in fk}
            controls = [x for x in gen if x["parent"] in parents]
            if controls and fk:
                y = [0] * len(controls) + [1] * len(fk)
                out[f"{g}/{ch}"] = metrics(y, [x["p"] for x in controls + fk])
    return out


def _rng(rid, ch):
    return np.random.default_rng(int(hashlib.sha256(f"{rid}/{ch}".encode()).hexdigest()[:8], 16))


def panel_rows(dev, names):
    """Deterministic development panels outside every decision fold."""
    from .stage0 import read

    rows = []
    if "libri-dev" in names:
        inv = [r for r in read(dev / "stage0" / "inventory.csv") if r["group"] in {"dev-clean", "dev-other"}]
        by = defaultdict(list)
        for r in inv:
            by[r["reader"]].append(r)
        for reader, v in sorted(by.items()):
            for r in sorted(v, key=lambda r: hashlib.sha256(r["uid"].encode()).hexdigest())[:LIBRI_PER_READER]:
                rows.append(dict(id="libri-dev/" + r["uid"], archive=r["archive"], member=r["member"],
                                 label="0", generator="genuine", panel="libri-dev/" + r["group"],
                                 speaker="librispeech/" + reader, voice=""))
    if "mlaad" in names:
        diag = read(dev / "stage0-en-ecapa-xtts" / "mlaad-diagnostic.csv")
        by = defaultdict(list)
        for r in diag:
            by[r["family"]].append(r)
        for fam, v in sorted(by.items()):
            for r in sorted(v, key=lambda r: hashlib.sha256(r["fake_member"].encode() + r["fake_archive"].encode()).hexdigest())[:MLAAD_PER_FAMILY]:
                reader = r["real_member"].split("/by_book/")[-1].split("/")[1]
                rows.append(dict(id="mlaad/" + r["fake_archive"], archive=r["fake_archive"], member=r["fake_member"],
                                 label="1", generator="mlaad/" + fam, panel="mlaad", speaker="mailabs/" + reader,
                                 voice=r["voice"], parent="mailabs/" + r["real_member"]))
                rows.append(dict(id="mailabs/" + r["real_member"], archive=r["real_archive"], member=r["real_member"],
                                 label="0", generator="genuine", panel="mlaad", speaker="mailabs/" + reader, voice="", parent="mailabs/" + r["real_member"]))
        seen, unique = set(), []
        for r in rows:
            if r["id"] not in seen:
                seen.add(r["id"])
                unique.append(r)
        rows = unique
    return rows


def matched_panel_scores(items):
    """Keep MLAAD parents and renditions together after four-second eligibility."""
    eligible = {(r["parent"], r["channel"], r["label"]) for r in items if r["panel"] == "mlaad"}
    return [r for r in items if r["panel"] != "mlaad"
            or (r["parent"], r["channel"], "1" if r["label"] == "0" else "0") in eligible]


def evaluate(cfg):
    import torch
    from scipy.special import expit

    from .audio import channel, decode, fixed_crop, model_view
    from .data import digest, entries, payload
    from .en_treatment import apply
    from .stage0 import groups as by_archive

    runs = cfg.data / "runs"
    frozen = json.loads((runs / "teacher-frozen.json").read_text())
    best = runs / "teacher.best.pt"
    if digest(best) != frozen["teacher_sha256"] or digest(cfg.local / "index.json") != frozen["index_sha256"]:
        raise ValueError("Teacher or pack changed since the threshold was frozen")
    audit = teacher_context(cfg)
    if frozen.get("exception") != audit.get("exception"):
        raise ValueError("Frozen exception differs from the bound audit")
    (a, b), (lo, hi) = frozen["platt"], frozen["thresholds"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_teacher(cfg, best, device, audit)

    def judge(raw, row, chans):
        res = []
        for ch in chans:
            x = model_view(channel(raw, ch, _rng(row["id"], ch)), {"pair": "phonemedf"})
            s = mean_logit(model, x, device)
            if s is None:
                continue
            p = float(expit(a * s + b))
            res.append(dict(id=row["id"], parent=row.get("parent", row["id"]), label=row["label"], generator=row["generator"], speaker=row["speaker"],
                            voice=row.get("voice", ""), channel=ch, p=p, verdict=verdict(p, lo, hi),
                            panel=row.get("panel", "validate")))
        return res

    items, skipped = [], defaultdict(int)
    with torch.inference_mode():
        for r in entries(cfg.local, {"validate"}):
            got = judge(decode(payload(cfg.local, r)), r, CHANNELS)
            skipped["validate"] += not got
            items += got
        panels = panel_rows(cfg.dev, cfg.panels) if cfg.panels else []
        if panels:
            from .data import SPLITS
            fold_readers = {r["speaker"] for r in entries(cfg.local, SPLITS)}
            if any(r["speaker"] in fold_readers for r in panels):
                raise ValueError("Development panel reader also occurs in decision folds")
        from .adapters import selected_members

        archives = by_archive(cfg.dev / "development", panels) if panels else {}
        for r, blob in selected_members(archives):
            crop = fixed_crop(blob)
            if crop is None:
                skipped[r["panel"]] += 1
                continue
            items += judge(decode(apply(crop, "norm")), r, PANEL_CHANNELS)
    matched = matched_panel_scores(items)
    unmatched = len(items) - len(matched)
    items = matched
    val = [x for x in items if x["panel"] == "validate"]
    pan = [x for x in items if x["panel"] != "validate"]
    unmeasured = dict(UNMEASURED)
    for name in cfg.panels or []:
        if not any(x["panel"].split("/")[0] == name for x in pan):
            unmeasured[name] = "No eligible development panel clips were scored"
    if not val:
        raise ValueError("No eligible validation clips were scored")
    result = {
        "kind": "EXPLORATORY Teacher V1 baseline evaluation" if frozen.get("exception") else ("Teacher V2-A evaluation" if audit.get("version") == "v2a-norm-1" else "Teacher V1 evaluation"),
        "exception": frozen.get("exception"),
        "frozen": frozen,
        "validate_by_generator_channel": summarise(val, lambda x: (x["generator"], x["channel"])),
        "validate_separability": separability(val),
        "panels_by_generator_channel": summarise(pan, lambda x: (x["panel"], x["generator"], x["channel"])),
        "false_alarms_on_genuine_by_channel": failures(items, "0"),
        "missed_fakes_by_generator": failures(items, "1"),
        "panels_requested": list(cfg.panels or []),
        "skipped_shorter_than_4s": dict(skipped),
        "mlaad_unpaired_channel_scores_excluded": unmatched,
        "unmeasured": unmeasured,
        "notes": [
            "Validate readers and voices are disjoint from train; Chatterbox never trained.",
            "Channel effects are simulated from clean development audio.",
            "All inputs use the training preprocessing: 4 s fixed crop, frozen norm.",
            "Sealed sets were not touched.",
        ],
    }
    clips = runs / "teacher-eval-clips.csv"
    tmp = clips.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0]))
        w.writeheader()
        w.writerows(items)
    tmp.replace(clips)
    output = runs / "teacher-eval.json"
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["freeze", "evaluate"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--local", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--dev", type=Path, help="DATA root holding development/ and stage0/ for panels")
    p.add_argument("--panels", nargs="*", choices=["libri-dev", "mlaad"], default=[])
    cfg = p.parse_args()
    out = freeze(cfg) if cfg.action == "freeze" else evaluate(cfg)
    print(json.dumps(out, indent=2, default=str)[:20000])


if __name__ == "__main__":
    main()
