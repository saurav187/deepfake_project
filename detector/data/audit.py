"""Merge source manifests into one audited manifest with splits and exclusions.

Usage (repo root):  venv/bin/python -m detector.data.audit

Rules (each is a lesson from the legacy pipeline):
  1. Undecodable files, or files with under MIN_SPEECH_S of speech, are excluded.
  2. Exact decoded-audio duplicates: if copies disagree on the label, all are
     excluded; if any copy is test data, the train/dev copies are excluded;
     otherwise one copy is kept.
  3. Sources without official partitions are split by speaker group (seeded).
  4. In train and in dev separately, only format strata (container, codec,
     rate, channels) containing both labels are kept, then downsampled to one
     common fake fraction, so the stored format cannot encode the label.
  5. Gate: format metadata must not predict the label in train or dev (AUC of
     each stratum's fake fraction within 0.5 +/- 0.10). Otherwise exit 1.
     Duration/silence/level shortcuts are measured and reported as warnings.
"""
from collections import defaultdict
import csv
import json
import random
import re
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from detector import paths

MIN_SPEECH_S = 0.5
DEV_FRACTION = 0.2
SPLIT_SEED = 2026
MIN_PER_LABEL_IN_STRATUM = 20
FORMAT_AUC_LIMIT = 0.60
SHORTCUT_AUC_WARN = 0.70
FORMAT = ["container", "subtype", "rate", "channels"]
SHORTCUT = ["stored_duration_s", "speech_s", "lead_silence_s", "trail_silence_s", "peak_dbfs", "active_dbfs"]


def load():
    rows = []
    for path in sorted(paths.MANIFESTS.glob("*.csv")):
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        r["excluded"] = ""
    return rows


def exclude(row, reason):
    if not row["excluded"]:
        row["excluded"] = reason


def assign_splits(rows):
    groups = defaultdict(set)
    for r in rows:
        if r["role"] == "unassigned":
            groups[r["source"]].add(r["group"])
    plan = {}
    for source, names in groups.items():
        named = sorted(g for g in names if re.search(r"Speaker-\d+$", g))
        for g in names - set(named):
            plan[g] = None  # e.g. an unspecified/mixed speaker folder cannot be split safely
        random.Random(SPLIT_SEED).shuffle(named)
        n_dev = max(1, round(DEV_FRACTION * len(named)))
        plan.update({g: "dev" for g in named[:n_dev]})
        plan.update({g: "train" for g in named[n_dev:]})
    for r in rows:
        if r["role"] == "unassigned":
            role = plan[r["group"]]
            if role is None:
                exclude(r, "speaker group not identifiable")
            else:
                r["role"] = role
    return plan


def dedupe(rows):
    by_hash = defaultdict(list)
    for r in rows:
        if not r["excluded"]:
            by_hash[r["pcm_sha256"]].append(r)
    stats = defaultdict(int)
    for copies in by_hash.values():
        if len(copies) < 2:
            continue
        if len({c["label"] for c in copies}) > 1:
            for c in copies:
                exclude(c, "duplicate audio with conflicting labels")
            stats["conflicting_label_sets"] += 1
            continue
        if any(c["role"] == "test" for c in copies):
            for c in copies:
                if c["role"] != "test":
                    exclude(c, "duplicate of test audio")
                    stats["train_dev_copies_of_test"] += 1
            keep = [c for c in copies if c["role"] == "test"]
        else:
            keep = copies
            if len({c["role"] for c in copies}) > 1:
                stats["cross_split_sets"] += 1
        keep = sorted(keep, key=lambda c: c["uid"])
        for c in keep[1:]:
            exclude(c, "duplicate audio")
            stats["duplicates_removed"] += 1
        if len({c["group"] for c in copies}) > 1:
            stats["sets_spanning_speaker_groups"] += 1
    return dict(stats)


def balance_formats(rows):
    strata = defaultdict(lambda: [0, 0])
    for r in rows:
        if not r["excluded"] and r["role"] in ("train", "dev"):
            strata[(r["source"], r["role"], *[r[k] for k in FORMAT])][r["label"]] += 1
    kept = {k for k, (real, fake) in strata.items()
            if min(real, fake) >= MIN_PER_LABEL_IN_STRATUM}
    for r in rows:
        if not r["excluded"] and r["role"] in ("train", "dev"):
            if (r["source"], r["role"], *[r[k] for k in FORMAT]) not in kept:
                exclude(r, "format stratum lacks one label")

    # Both labels present is not enough: strata with different fake fractions
    # still let the format predict the label. Within each source and role, downsample so
    # every kept stratum has the same fake fraction p, choosing p to keep the
    # most recordings. Format is then statistically independent of the label.
    targets = {}
    for scope in {k[:2] for k in kept}:
        cells = [strata[k] for k in kept if k[:2] == scope]
        def kept_total(p):
            return sum(min(g, f * (1 - p) / p) + min(f, g * p / (1 - p)) for g, f in cells)
        targets[scope] = max(np.linspace(0.05, 0.95, 181), key=kept_total)
    members = defaultdict(lambda: ([], []))
    for r in rows:
        key = (r["source"], r["role"], *[r[k] for k in FORMAT])
        if not r["excluded"] and r["role"] in ("train", "dev") and key in kept:
            members[key][r["label"]].append(r)
    quota = {}
    for key, (genuine, fake) in members.items():
        p = targets[key[:2]]
        n_g = min(len(genuine), int(len(fake) * (1 - p) / p))
        n_f = min(len(fake), int(len(genuine) * p / (1 - p)))
        quota[key] = (n_g, n_f)
        for items, n in ((genuine, n_g), (fake, n_f)):
            items = sorted(items, key=lambda r: r["uid"])
            random.Random(f"{SPLIT_SEED}{key}").shuffle(items)
            for r in items[n:]:
                exclude(r, "format balancing (downsampled to equal fake fraction)")
    return [dict(source=k[0], role=k[1], **dict(zip(FORMAT, k[2:])), genuine=v[0], fake=v[1], kept=k in kept,
                 target_fake_fraction=round(float(targets[k[:2]]), 3) if k in kept else None,
                 kept_genuine=quota.get(k, (0, 0))[0], kept_fake=quota.get(k, (0, 0))[1])
            for k, v in sorted(strata.items(), key=lambda kv: -sum(kv[1]))]


def stratum_auc(rows, features):
    """In-sample AUC of each format stratum's own fake fraction as the score.

    With a handful of categorical strata there is nothing to overfit, and this
    avoids the anti-correlation that cross-validation produces on balanced data.
    0.5 means the format carries no information about the label."""
    y = np.array([r["label"] for r in rows])
    if len(set(y)) < 2:
        return None
    keys = [tuple(r[k] for k in features) for r in rows]
    totals = defaultdict(lambda: [0, 0])
    for k, label in zip(keys, y):
        totals[k][0] += label
        totals[k][1] += 1
    score = np.array([totals[k][0] / totals[k][1] for k in keys])
    return float(roc_auc_score(y, score))


def grouped_auc(rows, features, categorical):
    """Out-of-fold AUC of a metadata-only classifier, folds split by speaker group."""
    y = np.array([r["label"] for r in rows])
    if len(set(y)) < 2:
        return None
    groups = np.array([r["group"] for r in rows])
    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        return None
    cols = []
    for k in features:
        values = [r[k] for r in rows]
        if k in categorical:
            codes = {v: i for i, v in enumerate(sorted(set(values)))}
            cols.append([codes[v] for v in values])
        else:
            cols.append([float(v) if v not in ("", None) else np.nan for v in values])
    X = np.array(cols, dtype=float).T
    if np.all(np.nan_to_num(X) == np.nan_to_num(X[0])):
        return 0.5  # identical metadata for every file cannot carry the label
    # Average the AUC within each held-out fold. Pooling out-of-fold scores would
    # let differences in class balance between folds masquerade as signal.
    aucs, weights = [], []
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(set(y[train])) < 2 or len(set(y[test])) < 2:
            continue
        model = HistGradientBoostingClassifier(max_iter=100, categorical_features=[
            i for i, k in enumerate(features) if k in categorical] or None, random_state=0)
        model.fit(X[train], y[train])
        aucs.append(roc_auc_score(y[test], model.predict_proba(X[test])[:, 1]))
        weights.append(len(test))
    return float(np.average(aucs, weights=weights)) if aucs else None


def leakage(rows):
    report = []
    active = [r for r in rows if not r["excluded"] and not r["error"]]
    scopes = {f"{role} (all sources)": [r for r in active if r["role"] == role] for role in ("train", "dev")}
    for source in sorted({r["source"] for r in active}):
        for scope_roles, name in ((("train",), "train"), (("dev",), "dev"), (("test",), "test")):
            subset = [r for r in active if r["source"] == source and r["role"] in scope_roles]
            if subset:
                scopes[f"{source} {name}"] = subset
    for name, subset in scopes.items():
        report.append(dict(scope=name, n=len(subset), fake_fraction=round(np.mean([r["label"] for r in subset]), 4),
                           format_auc=stratum_auc(subset, FORMAT),
                           shortcut_auc=grouped_auc(subset, SHORTCUT, set()),
                           **{f"{k}_auc": grouped_auc(subset, [k], set()) for k in ("speech_s", "lead_silence_s", "trail_silence_s")}))
    return report


def counts(rows):
    table = defaultdict(lambda: [0, 0])
    for r in rows:
        if not r["excluded"]:
            table[(r["source"], r["role"])][r["label"]] += 1
    return [dict(source=s, role=role, genuine=v[0], fake=v[1],
                 groups=len({r["group"] for r in rows if not r["excluded"] and r["source"] == s and r["role"] == role}),
                 attacks=len({r["attack"] for r in rows if not r["excluded"] and r["source"] == s and r["role"] == role and r["label"]}))
            for (s, role), v in sorted(table.items())]


def main():
    rows = load()
    if not rows:
        sys.exit("No manifests found. Run: venv/bin/python -m detector.data.build_manifest")
    for r in rows:
        if r["error"]:
            exclude(r, f"decode: {r['error'][:60]}")
        elif float(r["speech_s"]) < MIN_SPEECH_S:
            exclude(r, f"less than {MIN_SPEECH_S}s of speech")
    split_plan = assign_splits(rows)
    duplicate_stats = dedupe(rows)
    strata = balance_formats(rows)
    leaks = leakage(rows)
    reasons = defaultdict(int)
    for r in rows:
        if r["excluded"]:
            reasons[(r["source"], r["excluded"])] += 1

    paths.AUDIT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (paths.AUDIT / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["uid"]))
    gates = [l for l in leaks if l["scope"].endswith("(all sources)")]
    passed = all(l["format_auc"] is None or abs(l["format_auc"] - 0.5) <= FORMAT_AUC_LIMIT - 0.5 for l in gates)
    report = dict(passed=passed, format_auc_limit=FORMAT_AUC_LIMIT, counts=counts(rows),
                  exclusions=[dict(source=s, reason=k, n=n) for (s, k), n in sorted(reasons.items())],
                  duplicates=duplicate_stats, format_strata=strata, leakage=leaks,
                  split_plan={g: r for g, r in sorted(split_plan.items())},
                  warnings=[f"{l['scope']}: duration/silence/level metadata predicts the label (AUC {l['shortcut_auc']:.3f})"
                            for l in leaks if l["shortcut_auc"] and l["shortcut_auc"] > SHORTCUT_AUC_WARN])
    (paths.AUDIT / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    print("\nRecordings kept (genuine / fake, speaker groups, attacks):")
    for c in report["counts"]:
        print(f"  {c['source']:12} {c['role']:6} {c['genuine']:>7} / {c['fake']:<7} groups {c['groups']:>4}  attacks {c['attacks']}")
    print("\nExclusions:")
    for e in report["exclusions"]:
        print(f"  {e['source']:12} {e['n']:>6}  {e['reason']}")
    print(f"\nDuplicates: {duplicate_stats}")
    print("\nMetadata-only label prediction (grouped-CV AUC; 0.5 = no leak):")
    fmt = lambda v: "  n/a" if v is None else f"{v:.3f}"
    for l in leaks:
        print(f"  {l['scope']:28} n={l['n']:>7}  format {fmt(l['format_auc'])}  duration/silence/level {fmt(l['shortcut_auc'])}")
    for w in report["warnings"]:
        print(f"WARNING {w}")
    print(f"\nFormat gate (train and dev format AUC within 0.5 +/- {FORMAT_AUC_LIMIT - 0.5:.2f}): {'PASS' if passed else 'FAIL'}")
    print(f"Wrote {paths.AUDIT / 'manifest.csv'} and report.json")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
