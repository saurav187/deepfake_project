"""Compare trained runs, then calibrate and freeze a release, using dev data only.

Compare runs (pick the encoder):
  venv/bin/python -m detector.calibrate --compare detector/runs/models/*

Calibrate one run, or an ensemble of seeds of the same encoder, into a release:
  venv/bin/python -m detector.calibrate --runs detector/runs/models/xlsr_s42 detector/runs/models/xlsr_s43 \
      --out detector/runs/release/v1

What calibration does:
  1. Recording score = mean window logit, averaged over the ensemble's seeds.
  2. Clean and MP3 dev scores are pooled, so thresholds hold for compressed uploads.
  3. Every (source, label) cell gets equal total weight, so ASVspoof 5's ~90% spoof
     share cannot skew the fit. The calibrated probability therefore assumes a
     50/50 prior between genuine and fake.
  4. Platt scaling: p = sigmoid(a * logit + b), fitted by weighted logistic regression.
  5. Verdict thresholds from explicit error budgets on the weighted dev data:
       likely_fake  if p >= t_fake  (at most FA_BUDGET of genuine audio flagged)
       likely_real  if p <  t_real  (at most MISS_BUDGET of fakes pass as real)
       uncertain    otherwise
The release is hashed (checkpoints and this file) so the one-time test evaluation
can verify that nothing changed after calibration.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from detector import paths
from detector.train.data import WINDOW_S, DevWindows

DEV_MAX_WINDOWS = DevWindows([]).max_windows

FA_BUDGET = 0.05
MISS_BUDGET = 0.05
MIN_SPEECH_S = 1.0
CHANNELS = ("clean", "mp3_32k")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def repo_relative(path):
    """Repo-relative path when inside the repository (portable), else absolute."""
    path = Path(path).resolve()
    return str(path.relative_to(paths.REPO)) if path.is_relative_to(paths.REPO) else str(path)


def eer(y, s, w=None):
    fpr, tpr, _ = roc_curve(y, s, sample_weight=w)
    i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def read_run(run):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None
    with (run / "dev_predictions.csv").open() as f:
        rows = list(csv.DictReader(f))
    return config, summary, rows


# ------------------------------------------------------------------ compare
def compare(runs):
    table = []
    for run in runs:
        run = Path(run)
        if not (run / "summary.json").exists():
            print(f"skip (unfinished): {run}")
            continue
        config, summary, _ = read_run(run)
        with (run / "history.csv").open() as f:
            history = list(csv.DictReader(f))
        best = min(history, key=lambda h: float(h["selection_eer"]))
        per = {k: float(v) for k, v in best.items() if k.endswith("/eer")}
        table.append(dict(run=run.name, encoder=config["encoder"], seed=config["seed"], best_epoch=int(best["epoch"]),
                          epochs=len(history), selection_eer=float(best["selection_eer"]), **per))
    if not table:
        raise SystemExit("No finished runs.")
    keys = sorted({k for t in table for k in t if k.endswith("/eer")})
    print(f"{'run':28}{'encoder':22}{'seed':>5}{'epoch':>6}{'mean EER':>10}" + "".join(f"{k[:-4]:>22}" for k in keys))
    for t in sorted(table, key=lambda t: t["selection_eer"]):
        print(f"{t['run']:28}{t['encoder']:22}{t['seed']:>5}{t['best_epoch']:>6}{t['selection_eer']:>10.4f}"
              + "".join(f"{t.get(k, float('nan')):>22.4f}" for k in keys))
    by_encoder = defaultdict(list)
    for t in table:
        by_encoder[t["encoder"]].append(t["selection_eer"])
    print("\nMean dev selection EER by encoder (lower is better):")
    for encoder, values in sorted(by_encoder.items(), key=lambda kv: np.mean(kv[1])):
        print(f"  {encoder:22} {np.mean(values):.4f}  over {len(values)} run(s)")


# ---------------------------------------------------------------- calibrate
def ensemble(runs):
    configs, members, logits = [], [], {}
    reference = None
    for run in runs:
        config, summary, rows = read_run(run)
        if summary is None:
            raise SystemExit(f"Run not finished: {run}")
        uids = [r["uid"] for r in rows]
        if reference is None:
            reference = rows
        elif uids != [r["uid"] for r in reference]:
            raise SystemExit(f"{run} scored a different dev set; ensembles need identical dev subsets.")
        for c in CHANNELS:
            logits.setdefault(c, []).append(np.array([float(r[f"logit_{c}"]) for r in rows]))
        configs.append(config)
        members.append(dict(run=repo_relative(run), seed=config["seed"],
                            best_epoch=summary["best_epoch"], checkpoint_sha256=sha256(Path(run) / "best.pt")))
    encoders = {c["encoder"] for c in configs}
    if len(encoders) != 1:
        raise SystemExit(f"An ensemble must use one encoder, got {encoders}")
    with np.errstate(invalid="ignore"):
        mean = {c: np.mean(np.stack(v), axis=0) for c, v in logits.items()}
    return encoders.pop(), members, reference, mean


def cell_weights(rows):
    """Weights giving every (source, label) cell the same total weight."""
    counts = defaultdict(int)
    for r in rows:
        counts[(r["source"], r["label"])] += 1
    return np.array([1.0 / counts[(r["source"], r["label"])] for r in rows])


def budget_thresholds(y, p, w):
    """t_fake: lowest p with weighted genuine flag rate <= FA_BUDGET.
    t_real: highest p with weighted fake pass rate (p < t_real) <= MISS_BUDGET."""
    genuine, fake = y == 0, y == 1
    order = np.sort(np.unique(p))
    fa = np.array([w[genuine & (p >= t)].sum() / w[genuine].sum() for t in order])
    miss = np.array([w[fake & (p < t)].sum() / w[fake].sum() for t in order])
    t_fake = float(order[np.argmax(fa <= FA_BUDGET)]) if (fa <= FA_BUDGET).any() else 1.0
    ok = np.flatnonzero(miss <= MISS_BUDGET)
    t_real = float(order[ok[-1]]) if len(ok) else 0.0
    return min(t_real, t_fake), t_fake


def describe(y, p, w, t_real, t_fake):
    genuine, fake = y == 0, y == 1
    frac = lambda mask, cls: float(w[mask & cls].sum() / w[cls].sum()) if w[cls].sum() else None
    return dict(n=int(len(y)), auc=float(roc_auc_score(y, p, sample_weight=w)) if len(set(y)) == 2 else None,
                eer=eer(y, p, w) if len(set(y)) == 2 else None,
                genuine_flagged_fake=frac(p >= t_fake, genuine), genuine_uncertain=frac((p >= t_real) & (p < t_fake), genuine),
                genuine_called_real=frac(p < t_real, genuine), fake_detected=frac(p >= t_fake, fake),
                fake_uncertain=frac((p >= t_real) & (p < t_fake), fake), fake_missed_as_real=frac(p < t_real, fake))


def calibrate(runs, out):
    encoder, members, rows, logits = ensemble(runs)
    # Pool channels: each recording appears once per channel.
    data = [(r, c, logits[c][i]) for c in CHANNELS for i, r in enumerate(rows) if np.isfinite(logits[c][i])]
    pooled = [dict(source=r["source"], label=int(r["label"])) for r, _, _ in data]
    y = np.array([d["label"] for d in pooled])
    x = np.array([s for _, _, s in data]).reshape(-1, 1)
    w = cell_weights(pooled)
    w = w / w.mean()
    platt = LogisticRegression(C=1e6).fit(x, y, sample_weight=w)  # near-unregularised Platt scaling
    a, b = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    if a <= 0:
        raise SystemExit(f"Calibration slope {a:.3f} <= 0: the model does not rank dev audio usefully.")
    p = 1 / (1 + np.exp(-(a * x[:, 0] + b)))
    t_real, t_fake = budget_thresholds(y, p, w)

    report = {"pooled (source-balanced)": describe(y, p, w, t_real, t_fake)}
    for source in sorted({d["source"] for d in pooled}):
        for channel in CHANNELS:
            m = np.array([d["source"] == source and c == channel for d, (_, c, _) in zip(pooled, data)])
            report[f"{source} / {channel}"] = describe(y[m], p[m], np.ones(m.sum()), t_real, t_fake)
    attacks = defaultdict(list)
    for (r, c, _), prob in zip(data, p):
        if r["label"] == "1" and c == "clean":
            attacks[(r["source"], r["attack"])].append(prob >= t_fake)
    per_attack = {f"{s}/{a}": float(np.mean(v)) for (s, a), v in sorted(attacks.items())}

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    release = dict(
        name=out.name, created=datetime.now(UTC).isoformat(timespec="seconds"), encoder=encoder,
        members=members, aggregation="mean window logit per recording, then mean over members",
        window_s=WINDOW_S, dev_max_windows_per_recording=DEV_MAX_WINDOWS,
        note=("Dev scores (and so this calibration) use at most DEV_MAX_WINDOWS evenly spaced windows per "
              "recording; that covers recordings up to 16 s completely. Inference averages every window."),
        calibration=dict(method="platt", a=a, b=b, prior="50/50 (source- and label-balanced dev)",
                         channels=list(CHANNELS)),
        thresholds=dict(likely_real_below=t_real, likely_fake_at_or_above=t_fake,
                        fa_budget=FA_BUDGET, miss_budget=MISS_BUDGET, min_speech_s=MIN_SPEECH_S),
        dev=report, dev_fake_detected_by_attack_clean=per_attack,
        audit_report_sha256=sha256(paths.AUDIT / "report.json"),
        audit_manifest_sha256=sha256(paths.AUDIT / "manifest.csv"),
        calibrate_script_sha256=sha256(Path(__file__)),
    )
    (out / "release.json").write_text(json.dumps(release, indent=2) + "\n")
    (out / "release.sha256").write_text(sha256(out / "release.json") + "  release.json\n")

    print(f"Release {out.name}: {encoder}, {len(members)} member(s)")
    print(f"Platt: p = sigmoid({a:.4f} * logit + {b:+.4f})")
    print(f"Thresholds: likely_real < {t_real:.4f} <= uncertain < {t_fake:.4f} <= likely_fake")
    print(f"\n{'dev slice':34}{'n':>7}{'AUC':>7}{'EER':>7}  genuine: fake/unc/real   fake: fake/unc/real")
    for name, d in report.items():
        f = lambda v: f"{v:5.1%}" if v is not None else "  n/a"
        auc = f"{d['auc']:7.3f}" if d["auc"] is not None else "    n/a"
        e = f"{d['eer']:7.3f}" if d["eer"] is not None else "    n/a"
        print(f"{name:34}{d['n']:>7}{auc}{e}  {f(d['genuine_flagged_fake'])}/{f(d['genuine_uncertain'])}/{f(d['genuine_called_real'])}"
              f"  {f(d['fake_detected'])}/{f(d['fake_uncertain'])}/{f(d['fake_missed_as_real'])}")
    print(f"\nWrote {out / 'release.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", nargs="+", metavar="RUN", help="summarise finished runs")
    p.add_argument("--runs", nargs="+", metavar="RUN", help="runs (seeds of one encoder) to calibrate")
    p.add_argument("--out", type=Path, help="new release directory")
    args = p.parse_args()
    if args.compare:
        compare(args.compare)
    elif args.runs and args.out:
        calibrate(args.runs, args.out)
    else:
        p.error("use --compare RUN... or --runs RUN... --out DIR")


if __name__ == "__main__":
    main()
