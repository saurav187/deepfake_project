"""Modern-TTS 70%-at-15% experiment: release-v1 fallback x exploratory Teacher V1.

CPU-only analysis of already-frozen per-recording scores. Nothing is retrained and
no audio is read. Inputs (all read-only):

  <data>/v2/day-target-1/fallback-detector-v1/scores/<sha256(id)>.json   release v1
  <data>/stage0-en-ecapa-xtts-norm-exploratory/runs/teacher-eval-clips.csv  Teacher V1

Output goes only to <data>/v2/modern-tts-70-at-15/ (or --out).

Phases
  overlap   descriptive, in-sample: which clips the teacher adds at the 15% limit,
            the extra genuine false alarms, and the OR-rule cost.
  crossfit  prespecified score combination, reader-disjoint fit / calibrate /
            evaluate rotation inside the three-reader MLAAD development panel.

The three-reader MLAAD panel has been inspected repeatedly; every number produced
here is development evidence and cannot certify the 70%/15% target.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

FAMILIES = ["Chatterbox", "ElevenLabs-v3", "FishTTS", "Gemini-3.1-Flash-TTS"]
TEACHER_CSV = "stage0-en-ecapa-xtts-norm-exploratory/runs/teacher-eval-clips.csv"
FALLBACK = "v2/day-target-1/fallback-detector-v1"
TARGET_DET = 0.70
FA_LIMIT = 0.15
# Prespecified before any combined score was computed (see protocol in report).
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]  # weight on the release fallback z-score
EPS = 1e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def load(root, channel="clean", view="paired"):
    """One row per clean MLAAD recording with both models' scores."""
    rows = []
    with (root / TEACHER_CSV).open() as f:
        for r in csv.DictReader(f):
            if r["panel"] != "mlaad" or r["channel"] != channel:
                continue
            fam = "genuine" if r["label"] == "0" else r["generator"].removeprefix("mlaad/")
            if fam not in FAMILIES + ["genuine"]:
                raise ValueError(f"unexpected family {fam}")
            key = hashlib.sha256(r["id"].encode()).hexdigest()
            fb = json.loads((root / FALLBACK / "scores" / f"{key}.json").read_text())
            if fb.get("_overlap"):
                raise ValueError(f"fallback flagged training overlap for {r['id']}")
            rows.append(
                {
                    "id": r["id"],
                    "parent": r["parent"],
                    "reader": r["speaker"],
                    "family": fam,
                    "label": int(r["label"]),
                    "teacher": float(r["p"]),
                    "release": float(fb[channel][view]["p"]),
                }
            )
    ids = Counter(r["id"] for r in rows)
    if any(v > 1 for v in ids.values()):
        raise ValueError("duplicate recording ids")
    return rows


def cutoff_at_fa(genuine_scores, limit=FA_LIMIT):
    """Lowest cutoff (score >= cutoff flags) with at most floor(limit*n) genuine flagged."""
    s = sorted(genuine_scores, reverse=True)
    allowed = int(len(s) * limit)
    return math.nextafter(s[allowed], math.inf) if allowed < len(s) else -math.inf


def table(rows, flag):
    out = {"genuine": [0, 0]}
    out.update({f: [0, 0] for f in FAMILIES})
    for r in rows:
        cell = out[r["family"]]
        cell[0] += bool(flag(r))
        cell[1] += 1
    fakes = [out[f] for f in FAMILIES]
    out["pooled_fake"] = [sum(a for a, _ in fakes), sum(b for _, b in fakes)]
    out["min_family_rate"] = min(a / b for a, b in fakes)
    out["meets_target"] = (
        out["genuine"][0] <= FA_LIMIT * out["genuine"][1]
        and all(a >= TARGET_DET * b for a, b in fakes)
    )
    return out


def need(n):
    return math.ceil(TARGET_DET * n - 1e-9)


# ---------------------------------------------------------------- overlap phase
def overlap(rows):
    gen = [r for r in rows if r["label"] == 0]
    c_rel = cutoff_at_fa([r["release"] for r in gen])
    c_tea = cutoff_at_fa([r["teacher"] for r in gen])
    rel = lambda r: r["release"] >= c_rel
    tea = lambda r: r["teacher"] >= c_tea
    out = {
        "cutoffs": {"release": c_rel, "teacher": c_tea},
        "release_at_15": table(rows, rel),
        "teacher_at_15": table(rows, tea),
        "or_rule_both_at_15": table(rows, lambda r: rel(r) or tea(r)),
        "and_rule_both_at_15": table(rows, lambda r: rel(r) and tea(r)),
        "by_family": {},
    }
    for fam in FAMILIES + ["genuine"]:
        rs = [r for r in rows if r["family"] == fam]
        both = sum(rel(r) and tea(r) for r in rs)
        rel_only = sum(rel(r) and not tea(r) for r in rs)
        tea_only = sum(tea(r) and not rel(r) for r in rs)
        neither = sum(not rel(r) and not tea(r) for r in rs)
        d = {
            "n": len(rs),
            "both": both,
            "release_only": rel_only,
            "teacher_only": tea_only,
            "neither": neither,
        }
        if fam != "genuine":
            d["needed_for_70"] = need(len(rs))
            d["release_gap"] = max(0, need(len(rs)) - (both + rel_only))
            d["teacher_only_covers_gap"] = tea_only >= d["release_gap"]
            # Readers of missed clips, to see if gains concentrate on one reader.
            d["teacher_only_by_reader"] = dict(
                Counter(r["reader"] for r in rs if tea(r) and not rel(r))
            )
        out["by_family"][fam] = d
    # Rank agreement among genuine and per family (Spearman) as a complementarity hint.
    out["spearman"] = {
        fam: spearman(
            [r["release"] for r in rows if r["family"] == fam],
            [r["teacher"] for r in rows if r["family"] == fam],
        )
        for fam in FAMILIES + ["genuine"]
    }
    # OR rule with a shared tightening: both models at the same per-model FA level q,
    # smallest union FA <= 15%. In-sample, descriptive only.
    best = None
    for k in range(int(len(gen) * FA_LIMIT), -1, -1):
        q = k / len(gen)
        cr = cutoff_at_fa([r["release"] for r in gen], q)
        ct = cutoff_at_fa([r["teacher"] for r in gen], q)
        t = table(rows, lambda r: r["release"] >= cr or r["teacher"] >= ct)
        if t["genuine"][0] <= FA_LIMIT * len(gen):
            best = {"per_model_fa": q, "cutoffs": [cr, ct], **t}
            break
    out["or_rule_union_capped_at_15_in_sample"] = best
    return out


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


# --------------------------------------------------------------- crossfit phase
def normaliser(genuine_rows):
    """z-normalise logit scores by genuine mean/sd of the fit fold (per model)."""
    stats = {}
    for m in ("release", "teacher"):
        v = [logit(r[m]) for r in genuine_rows]
        stats[m] = (statistics.fmean(v), statistics.pstdev(v) or 1.0)
    return stats


def combined(r, stats, w):
    zr = (logit(r["release"]) - stats["release"][0]) / stats["release"][1]
    zt = (logit(r["teacher"]) - stats["teacher"][0]) / stats["teacher"][1]
    return w * zr + (1 - w) * zt


def select_weight(fit_rows, stats):
    """Weight maximising the worst-family detection at the fit fold's own 15% cutoff.

    Ties -> larger pooled detection -> weight closest to 0.5 (prespecified)."""
    best = None
    for w in WEIGHTS:
        s = {r["id"]: combined(r, stats, w) for r in fit_rows}
        c = cutoff_at_fa([s[r["id"]] for r in fit_rows if r["label"] == 0])
        t = table(fit_rows, lambda r: s[r["id"]] >= c)
        key = (t["min_family_rate"], t["pooled_fake"][0], -abs(w - 0.5))
        if best is None or key > best[0]:
            best = (key, w, t)
    return best[1], best[2]


def crossfit(rows):
    readers = sorted({r["reader"] for r in rows})
    if len(readers) != 3:
        raise ValueError(f"expected 3 readers, got {readers}")
    by = {k: [r for r in rows if r["reader"] == k] for k in readers}
    folds = []
    pooled = defaultdict(lambda: [0, 0])
    for fit, cal, ev in itertools.permutations(readers):
        stats = normaliser([r for r in by[fit] if r["label"] == 0])
        w, fit_table = select_weight(by[fit], stats)
        cal_gen = [combined(r, stats, w) for r in by[cal] if r["label"] == 0]
        thr = cutoff_at_fa(cal_gen)
        flag = lambda r: combined(r, stats, w) >= thr
        ev_table = table(by[ev], flag)
        # Same protocol for each single model (threshold frozen on cal reader).
        singles = {}
        for m in ("release", "teacher"):
            tm = cutoff_at_fa([r[m] for r in by[cal] if r["label"] == 0])
            singles[m] = table(by[ev], lambda r, m=m, tm=tm: r[m] >= tm)
        folds.append(
            {
                "fit_reader": fit,
                "calibrate_reader": cal,
                "evaluate_reader": ev,
                "normaliser": stats,
                "weight_release": w,
                "fit_fold_table": fit_table,
                "threshold": thr,
                "evaluate": ev_table,
                "single_models_same_protocol": singles,
            }
        )
        for k in ["genuine"] + FAMILIES:
            pooled[k][0] += ev_table[k][0]
            pooled[k][1] += ev_table[k][1]
    return {
        "design": (
            "6 ordered (fit, calibrate, evaluate) reader permutations of the 3-reader "
            "MLAAD development panel; each evaluate reader appears in 2 rotations, so "
            "pooled counts double-count clips (each clip evaluated twice)"
        ),
        "readers": readers,
        "clips_per_reader": {
            k: dict(Counter(r["family"] for r in by[k])) for k in readers
        },
        "folds": folds,
        "pooled_over_rotations": {k: list(v) for k, v in pooled.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["overlap", "crossfit", "all"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--view", default="paired", choices=["paired", "full"])
    a = p.parse_args()
    out = a.out or a.data / "v2/modern-tts-70-at-15"
    out.mkdir(parents=True, exist_ok=True)
    rows = load(a.data, view=a.view)
    prov = {
        "teacher_csv_sha256": sha(a.data / TEACHER_CSV),
        "fallback_result_sha256": sha(a.data / FALLBACK / "result.json"),
        "fallback_inputs_sha256": sha(a.data / FALLBACK / "inputs.json"),
        "fallback_view": a.view,
        "channel": "clean",
        "rows": dict(Counter(r["family"] for r in rows)),
        "readers": dict(Counter(r["reader"] for r in rows)),
    }
    phases = ["overlap", "crossfit"] if a.phase == "all" else [a.phase]
    for ph in phases:
        res = {"provenance": prov, **(overlap(rows) if ph == "overlap" else crossfit(rows))}
        path = out / f"{ph}-{a.view}.json"
        path.write_text(json.dumps(res, indent=2) + "\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
