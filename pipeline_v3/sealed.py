"""Custodian-only evaluation of a frozen release. No entry point in development notebooks."""

import argparse
import csv
import json
import tempfile
from pathlib import Path

from .data import digest
from .metrics import bootstrap, metrics
from .release import Detector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--sealed-root", type=Path, required=True)
    p.add_argument(
        "--set",
        choices=[
            "itw",
            "va",
            "voxenes",
            "deepfake-eval",
            "owner-private",
            "app-testing",
            "llamapartial",
        ],
        required=True,
    )
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument(
        "--expected-release-sha256",
        required=True,
        help="Pre-registered hash of release.json",
    )
    p.add_argument("--dry-run", action="store_true")
    cfg = p.parse_args()
    version = digest(cfg.release / "release.json")
    if version != cfg.expected_release_sha256:
        raise ValueError("Release differs from preregistration")
    if cfg.dry_run:
        print(json.dumps({"release": version, "confirmation": cfg.set}))
        return
    detector = Detector(cfg.release)
    root = (cfg.sealed_root / cfg.set).resolve()
    cfg.ledger.mkdir(parents=True, exist_ok=True)
    dest = cfg.ledger / (version + "-" + cfg.set + ".json")
    # Exclusive reservation prevents a second evaluation, including after a partial run.
    with dest.open("x") as marker:
        json.dump({"status": "started", "set": cfg.set, "release": version}, marker)
    records = []
    outcomes = {
        str(l): dict.fromkeys(
            ["likely_real", "uncertain", "likely_fake", "insufficient_audio"], 0
        )
        for l in (0, 1)
    }
    with (root / "trials.csv").open() as f:
        for r in csv.DictReader(f):
            path = (root / r["path"]).resolve()
            if (
                not path.is_relative_to(root)
                or r["label"] not in {"0", "1"}
                or not r["speaker"]
            ):
                raise ValueError("Invalid custodian trial")
            result = detector.analyze(path.read_bytes(), "")
            outcomes[r["label"]][result["verdict"]] += 1
            if result["fake_probability"] is not None:
                records.append((r, result["fake_probability"]))
    rows = [r for r, _ in records]
    scores = [s for _, s in records]
    report = {
        "status": "complete",
        "release": version,
        "set": cfg.set,
        "outcomes": outcomes,
        "trial_sha256": digest(root / "trials.csv"),
        "discrimination": metrics([int(r["label"]) for r in rows], scores),
        "uncertainty": bootstrap(rows, scores),
    }
    with tempfile.NamedTemporaryFile(mode="w", dir=cfg.ledger, delete=False) as f:
        json.dump(report, f, indent=2)
    Path(f.name).replace(dest)
    print(dest)


if __name__ == "__main__":
    main()
