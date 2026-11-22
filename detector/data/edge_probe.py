"""Gate: after the shared prepare() step, recording edges must not predict the label.

Usage (repo root):  venv/bin/python -m detector.data.edge_probe [--per-label 1500]

ASVspoof 5 train has a silence shortcut: genuine files carry leading and trailing
silence (median 0.16 s / 0.22 s), fakes have none. prepare() trims edges with no
margin, but trimming relies on an energy threshold, so this probe measures what
is actually left. For a sample of train and dev recordings per source and label
it computes, after prepare():
  - onset / tail level: RMS of the first / last 0.25 s relative to the median
    active-frame level (dB), and
  - onset / tail quiet fraction: share of 20 ms frames in the first / last 0.5 s
    more than 30 dB below that level.
AUCs are orientation-free, max(AUC, 1 - AUC), within every source and role.
Quiet-frame features (the silence shortcut) must stay <= LIMIT. Speech level
differs between genuine and synthetic audio throughout a recording, not only at
its edges (ASVspoof 5 train: middle-window level AUC 0.54, just inside the onset
0.62), so edge level is judged against the same measure taken mid-recording.
Edge level above LEVEL_HARD fails; more than LEVEL_EXCESS above the middle is a
logged warning. With the current prepare(), ASVspoof 5 train tail level is 0.615
vs 0.537 mid-recording (excess 0.078), consistent with sentence-final declination
in genuine speech, which cannot be trimmed away without removing speech. Training
randomly fades both edges of both classes, and the trained model's scores on
end-of-recording vs interior windows are checked on dev. Otherwise exit 1.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from detector import paths
from detector.audio import SR, decode, frame_db, prepare
from detector.data.sources import read_bytes

LIMIT = 0.60           # silence features: absolute AUC limit
LEVEL_EXCESS = 0.07    # level features: allowed AUC above the same measure mid-recording
LEVEL_HARD = 0.70      # level features: absolute ceiling
EDGE_S, QUIET_S = 0.25, 0.5
FEATURES = ("onset_level_db", "tail_level_db", "onset_quiet_frac", "tail_quiet_frac", "middle_level_db")


def edge_features(row):
    try:
        wave = prepare(decode(read_bytes(row["locator"]))[0])
    except Exception:
        return None
    if len(wave) < 2 * int(QUIET_S * SR):
        return None
    db = frame_db(wave)
    active = np.median(db[db >= db.max() - 40])
    rms = lambda x: 20 * np.log10(max(np.sqrt(np.mean(x ** 2)), 1e-6))
    n_edge, n_quiet = int(EDGE_S * SR), int(QUIET_S * SR) // 320
    mid = len(wave) // 2
    return dict(onset_level_db=rms(wave[:n_edge]) - active, tail_level_db=rms(wave[-n_edge:]) - active,
                middle_level_db=rms(wave[mid - n_edge // 2: mid + n_edge // 2]) - active,
                onset_quiet_frac=float(np.mean(db[:n_quiet] < active - 30)),
                tail_quiet_frac=float(np.mean(db[-n_quiet:] < active - 30)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-label", type=int, default=1500, help="recordings sampled per source, role and label")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    with (paths.AUDIT / "manifest.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if not r["excluded"] and r["role"] in ("train", "dev")]
    rng = np.random.default_rng(11)
    cells = {}
    for r in rows:
        cells.setdefault((r["source"], r["role"], int(r["label"])), []).append(r)
    sample = []
    for key in sorted(cells):
        items = cells[key]
        sample.extend(items[i] for i in rng.permutation(len(items))[:args.per_label])
    with ThreadPoolExecutor(args.workers) as pool:
        feats = list(pool.map(edge_features, sample, buffersize=4 * args.workers))
    results, failed, warnings = [], [], []
    for source, role in sorted({(r["source"], r["role"]) for r in sample}):
        idx = [i for i, r in enumerate(sample) if r["source"] == source and r["role"] == role and feats[i]]
        y = np.array([int(sample[i]["label"]) for i in idx])
        if len(set(y)) < 2:
            failed.append(f"{source} {role}: one label missing")
            continue
        row = dict(source=source, role=role, n=len(idx))
        for k in FEATURES:
            v = np.array([feats[i][k] for i in idx])
            auc = roc_auc_score(y, v)
            row[k] = round(float(max(auc, 1 - auc)), 4)
            row[f"{k}_median_genuine"] = round(float(np.median(v[y == 0])), 3)
            row[f"{k}_median_fake"] = round(float(np.median(v[y == 1])), 3)
        for k in ("onset_quiet_frac", "tail_quiet_frac"):
            if row[k] > LIMIT:
                failed.append(f"{source} {role}: {k} AUC {row[k]:.3f} > {LIMIT}")
        for k in ("onset_level_db", "tail_level_db"):
            excess = row[k] - row["middle_level_db"]
            row[f"{k}_excess_over_middle"] = round(excess, 4)
            if row[k] > LEVEL_HARD:
                failed.append(f"{source} {role}: {k} AUC {row[k]:.3f} > {LEVEL_HARD}")
            elif excess > LEVEL_EXCESS:
                warnings.append(f"{source} {role}: {k} AUC {row[k]:.3f} exceeds the mid-recording level AUC "
                                f"{row['middle_level_db']:.3f} by {excess:.3f}")
        results.append(row)
    passed = not failed
    paths.AUDIT.mkdir(parents=True, exist_ok=True)
    (paths.AUDIT / "edge_probe.json").write_text(json.dumps(dict(passed=passed, limit=LIMIT, results=results,
                                                                 failures=failed, warnings=warnings), indent=2) + "\n")
    print(f"{'source':12}{'role':7}{'n':>6}" + "".join(f"{k:>18}" for k in FEATURES))
    for r in results:
        print(f"{r['source']:12}{r['role']:7}{r['n']:>6}" + "".join(f"{r[k]:>18.3f}" for k in FEATURES))
    for msg in warnings:
        print(f"EDGE WARNING {msg}")
    for msg in failed:
        print(f"EDGE GATE FAIL {msg}")
    print(f"Edge gate (quiet-frame AUC <= {LIMIT}; edge level AUC <= {LEVEL_HARD}, warning above middle + {LEVEL_EXCESS}): "
          f"{'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
