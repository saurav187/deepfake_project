"""One-time sealed evaluation of a frozen release on a test corpus.

Usage (repo root):
  venv/bin/python -m detector.eval.evaluate --release detector/runs/release/v1 \
      --source in_the_wild --out detector/runs/eval/v1_in_the_wild

Refuses to run unless the release hashes verify, the audit manifest is the one the
release was calibrated against, and every member checkpoint lives inside the
repository. Each (release, source) pair is evaluated once: runs are recorded in
detector/runs/sealed_ledger.jsonl and a repeat needs --rerun, which is logged.
Nothing here feeds back into training, selection or thresholds.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from detector import paths
from detector.audio import SR, decode, prepare_with_offset
from detector.data.sources import read_bytes
from detector.inference import Detector, sha256, windows_of

LEDGER = paths.RUNS / "sealed_ledger.jsonl"
BOOTSTRAP = 1000
STRATA = (("under 4 s", 0, 4), ("4-16 s", 4, 16), ("over 16 s", 16, 1e9))


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def rates(items):
    """Outcomes at the frozen thresholds; insufficient audio counts in the denominators."""
    out = {}
    for label, name in ((0, "genuine"), (1, "fake")):
        group = [r for r in items if r["label"] == label]
        n = len(group)
        counts = defaultdict(int)
        for r in group:
            counts[r["verdict"]] += 1
        out[name] = {"n": n, **{key: (counts[verdict] / n if n else None) for key, verdict in (
            ("likely_fake", "likely_fake"), ("uncertain", "uncertain"), ("likely_real", "likely_real"),
            ("insufficient", "insufficient_audio"))}}
    return out


def metrics(items):
    scored = [r for r in items if r["logit"] is not None]
    y = np.array([r["label"] for r in scored])
    s = np.array([r["logit"] for r in scored])
    p = np.array([r["p"] for r in scored])
    m = dict(n=len(items), scored=len(scored), outcomes=rates(items))
    if len(set(y)) == 2:
        m.update(auc=float(roc_auc_score(y, s)), eer=eer(y, s), brier=float(np.mean((p - y) ** 2)))
    return m


def bootstrap(items, seed=4242):
    """95% intervals resampling whole speaker groups (people are the independent unit)."""
    groups = defaultdict(list)
    for r in items:
        groups[r["group"]].append(r)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    for _ in range(BOOTSTRAP):
        sample = [r for k in rng.choice(keys, len(keys)) for r in groups[k]]
        m = metrics(sample)
        for k in ("auc", "eer"):
            if k in m:
                draws[k].append(m[k])
        for cls, key in (("genuine", "likely_fake"), ("fake", "likely_fake"), ("fake", "likely_real")):
            v = m["outcomes"][cls][key]
            if v is not None:
                draws[f"{cls}_{key}"].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--source", required=True, choices=["in_the_wild", "va"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rerun", action="store_true", help="allow a repeat evaluation (recorded in the ledger)")
    args = p.parse_args()
    release_dir = args.release.resolve()
    out = args.out.resolve()
    if out.exists():
        sys.exit(f"{out} exists; choose a new directory")

    det = Detector(release_dir)                       # verifies release and checkpoint hashes
    rel = det.release
    for member in rel["members"]:
        if Path(member["run"]).is_absolute():
            sys.exit("Sealed evaluation needs a release whose members live inside the repository.")
    if sha256(paths.AUDIT / "manifest.csv") != rel["audit_manifest_sha256"]:
        sys.exit("The audit manifest changed since calibration; test membership may differ. Refusing.")
    previous = []
    if LEDGER.exists():
        previous = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    if any(e["release_sha256"] == det.release_sha256 and e["source"] == args.source for e in previous) and not args.rerun:
        sys.exit(f"{args.source} was already evaluated with this release (see {LEDGER}); use --rerun to repeat.")

    with (paths.AUDIT / "manifest.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["source"] == args.source and r["role"] == "test" and not r["excluded"]]
    if not rows:
        sys.exit(f"No test rows for {args.source}")
    out.mkdir(parents=True)
    entry = dict(time=datetime.now(UTC).isoformat(timespec="seconds"), release=str(release_dir),
                 release_sha256=det.release_sha256, source=args.source, rows=len(rows), rerun=args.rerun)
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    def load(row):
        try:
            wave, _ = decode(read_bytes(row["locator"]))
            prepared, _ = prepare_with_offset(wave)
            if len(prepared) / SR < det.min_speech_s:
                return row, len(wave) / SR, len(prepared) / SR, None, "insufficient speech"
            windows, _ = windows_of(prepared)
            return row, len(wave) / SR, len(prepared) / SR, windows, ""
        except Exception as exc:
            return row, None, None, None, f"{type(exc).__name__}: {exc}"[:200]

    start = perf_counter()
    results, pending = [], []

    def flush():
        logits = det.window_logits(np.concatenate([w for *_, w in pending]))
        k = 0
        for row, duration, speech, w in pending:
            logit = float(np.mean(logits[k:k + len(w)]))
            k += len(w)
            prob = det.probability(logit)
            results.append(dict(uid=row["uid"], label=int(row["label"]), group=row["group"], attack=row["attack"],
                                duration_s=duration, speech_s=speech, windows=len(w), logit=logit, p=prob,
                                verdict=det.verdict(prob), error=""))
        pending.clear()

    with ThreadPoolExecutor(6) as pool:
        for i, (row, duration, speech, windows, error) in enumerate(pool.map(load, rows, buffersize=48), 1):
            if windows is None:
                results.append(dict(uid=row["uid"], label=int(row["label"]), group=row["group"], attack=row["attack"],
                                    duration_s=duration, speech_s=speech, windows=0, logit=None, p=None,
                                    verdict="insufficient_audio" if error == "insufficient speech" else "error",
                                    error=error))
            else:
                pending.append((row, duration, speech, windows))
                if sum(len(x[3]) for x in pending) >= 64:
                    flush()
            if i % 2000 == 0:
                print(f"{i}/{len(rows)} recordings, {perf_counter() - start:.0f}s", flush=True)
    if pending:
        flush()

    summary = dict(entry, elapsed_s=perf_counter() - start, encoder=rel["encoder"], members=len(rel["members"]),
                   thresholds=rel["thresholds"], overall=metrics(results), bootstrap_95=bootstrap(results),
                   errors=sum(r["verdict"] == "error" for r in results),
                   by_speech_length={name: metrics([r for r in results if r["speech_s"] is not None and lo <= r["speech_s"] < hi])
                                     for name, lo, hi in STRATA},
                   by_attack={a: metrics([r for r in results if r["attack"] in (a, "bonafide")])
                              for a in sorted({r["attack"] for r in results if r["label"] == 1})},
                   script_sha256=sha256(Path(__file__)), audit_manifest_sha256=rel["audit_manifest_sha256"])
    baseline = paths.RUNS / "baselines" / ("legacy_itw" if args.source == "in_the_wild" else "legacy_va") / "summary.csv"
    if baseline.exists():
        with baseline.open() as f:
            summary["legacy_baseline"] = next((r for r in csv.DictReader(f) if r["model"] == "audited_baseline"), None)
    with (out / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    o = summary["overall"]
    ci = summary["bootstrap_95"]
    print(f"\n{args.source}: {o['n']} recordings, {o['scored']} scored, {summary['errors']} errors, {summary['elapsed_s']:.0f}s")
    if "auc" in o:
        print(f"AUC {o['auc']:.4f} [{ci['auc'][0]:.4f}, {ci['auc'][1]:.4f}]   EER {o['eer']:.4f} [{ci['eer'][0]:.4f}, {ci['eer'][1]:.4f}]   Brier {o['brier']:.4f}")
    for cls in ("genuine", "fake"):
        c = o["outcomes"][cls]
        print(f"{cls:8} n={c['n']:>6}  likely_fake {c['likely_fake']:.1%}  uncertain {c['uncertain']:.1%}  "
              f"likely_real {c['likely_real']:.1%}  insufficient {c['insufficient']:.1%}")
    if summary.get("legacy_baseline"):
        b = summary["legacy_baseline"]
        print(f"legacy audited baseline: AUC {float(b['auc']):.4f}  EER {float(b['eer']):.4f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
