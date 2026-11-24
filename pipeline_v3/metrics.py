from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def metrics(y, scores):
    if len(set(y)) != 2:
        return {"n": len(y), "auc": None, "eer": None}
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    gap = fpr + tpr - 1
    i = np.searchsorted(gap, 0)
    eer = np.interp(0, gap[i - 1 : i + 1], fpr[i - 1 : i + 1])
    return {"n": len(y), "auc": float(roc_auc_score(y, scores)), "eer": float(eer)}


def slices(rows, scores):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        for field in ("language", "channel", "domain", "pair", "generator", "length"):
            if field in r:
                groups[(field, r[field])].append(i)
    reports = {}
    for (field, key), idx in groups.items():
        if field in {"domain", "generator"}:
            # Compare against genuine controls in matching populations.
            fakes = [i for i in idx if str(rows[i]["label"]) == "1"]
            if not fakes:
                continue
            cells = {
                (rows[i]["pair"], rows[i]["language"], rows[i]["channel"])
                for i in fakes
            }
            idx = fakes + [
                i
                for i, r in enumerate(rows)
                if str(r["label"]) == "0"
                and (r["pair"], r["language"], r["channel"]) in cells
            ]
        reports[f"{field}/{key}"] = metrics(
            [int(rows[i]["label"]) for i in idx], np.asarray(scores)[idx]
        )
    for lang in {r["language"] for r in rows if "language" in r}:
        fakes = [
            i
            for i, r in enumerate(rows)
            if r["language"] == lang and r["domain"] == "unseen" and r["label"] == "1"
        ]
        parents = {p for i in fakes for p in rows[i]["parent"].split("|")}
        controls = [
            i
            for i, r in enumerate(rows)
            if r["language"] == lang
            and r["label"] == "0"
            and parents.intersection(r["parent"].split("|"))
        ]
        idx = controls + fakes
        reports[f"language/{lang}/unseen"] = metrics(
            [int(rows[i]["label"]) for i in idx], np.asarray(scores)[idx]
        )
    return reports


def bootstrap(rows, scores, repeats=500):
    # A converted recording can share a source or target voice with another trial.
    # Resample their connected identity components to preserve both dependencies.
    roots = {}

    def find(person):
        while person != roots.setdefault(person, person):
            roots[person] = roots[roots[person]]
            person = roots[person]
        return person

    for r in rows:
        people = [
            v
            for f in ("speaker", "voice")
            for v in r.get(f, "").split("|")
            if v and v != "-"
        ]
        anchor = find(people[0])
        for person in people[1:]:
            roots[find(person)] = anchor
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[find(r["speaker"].split("|")[0])].append(i)
    keys = list(by)
    rng = np.random.default_rng(471)
    vals = []
    for _ in range(repeats):
        idx = [i for k in rng.choice(keys, len(keys), replace=True) for i in by[k]]
        result = metrics([int(rows[i]["label"]) for i in idx], np.asarray(scores)[idx])
        if result["eer"] is not None:
            vals.append(result["eer"])
    return {
        "speaker_clusters": len(keys),
        "valid_draws": len(vals),
        "eer_95": np.quantile(vals, [0.025, 0.975]).tolist()
        if vals and len(keys) > 1
        else None,
    }
