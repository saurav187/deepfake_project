"""Common command interface for Colab notebooks and terminal sessions."""

import argparse
import json
from pathlib import Path

from .data import SPLITS, digest, pack, stage


def bound_audit(work, index):
    from .audit import require_passing

    report = json.loads((work / "audit.json").read_text())
    if work.name == "v2l-lsv-norm-1" or report.get("version") == "v2l-lsv-norm-1":
        from .v2l import verify_published as verify_v2l

        verify_v2l(work, report, index)
        return report
    if work.name == "v2a-norm-1" or report.get("version") == "v2a-norm-1":
        from .v2a import verify_published

        verify_published(work, report, index)
        return report
    if report.get("pass") or "exception" not in report:
        require_passing(report, digest(index))
    else:
        # Owner-approved exploratory run on a failed audit (exploratory.py): the record
        # must match the failed report exactly; policy and pack binding still apply.
        from .audit import POLICY
        from .exploratory import exception_for

        prep = json.loads((work / "prep-audit.json").read_text())
        if (
            report.get("policy") != POLICY
            or report.get("index_sha256") != digest(index)
            or exception_for(work, prep) != report["exception"]
        ):
            raise ValueError("Exploratory exception is not bound to this pack")
    if "selection_sha256" in report or (work / "prep-audit.json").exists():
        prep = json.loads((work / "prep-audit.json").read_text())
        if (prep != {k: v for k, v in report.items() if k not in {"index_sha256", "exception"}}
                or digest(work / "selection.csv") != prep["selection_sha256"]
                or digest(work / "eligible.csv") != prep["eligible_sha256"]):
            raise ValueError("Preparation audit or selection changed since pack")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=[
            "plan",
            "inventory",
            "match",
            "catalog",
            "pack",
            "stage",
            "audit",
            "pilot",
            "teacher",
            "student",
            "export",
            "evaluate",
            "benchmark",
        ],
    )
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--local", type=Path, default=Path("/content/v3-local"))
    p.add_argument("--selection", default="selection.csv")
    p.add_argument("--catalog", default="catalog.csv")
    p.add_argument("--inventory", default="inventory.jsonl")
    p.add_argument("--heldout", nargs="*", default=["Gemini-3.1-Flash-TTS", "freevc24"])
    p.add_argument("--packed", type=Path)
    p.add_argument("--release", type=Path)
    p.add_argument("--encoder", type=Path)
    p.add_argument("--small-encoder", type=Path)
    p.add_argument(
        "--splits", nargs="+", choices=sorted(SPLITS), default=sorted(SPLITS)
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--per-class", type=int, default=1000)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--languages", nargs="+", choices=["en", "hi"], default=["en", "hi"],
        help="pilot languages; must equal the languages the bound audit covered",
    )
    cfg = p.parse_args()
    cfg.packed = cfg.packed or cfg.data / "packed"
    cfg.release = cfg.release or cfg.data / "release-v3"
    cfg.encoder = cfg.encoder or cfg.data / "models" / "xlsr"
    cfg.small_encoder = cfg.small_encoder or cfg.data / "models" / "wavlm-base"
    if cfg.dry_run or cfg.action == "plan":
        print(
            json.dumps(
                {
                    "action": cfg.action,
                    "data": str(cfg.data),
                    "stage": str(cfg.local),
                    "initial_download_GB": 35.409,
                    "selected_training_clips_per_class": cfg.per_class,
                    "pilot_L4_hours_estimate": [0.25, 1],
                    "teacher_A100_hours_estimate": [1, 3],
                    "student_L4_hours_estimate": [1, 3],
                    "ordered_actions": [
                        "inventory",
                        "match",
                        "catalog",
                        "pack",
                        "stage",
                        "audit",
                        "pilot",
                        "teacher",
                        "student",
                        "export",
                        "evaluate",
                        "benchmark",
                    ],
                }
            )
        )
        return
    if cfg.action == "inventory":
        from .adapters import inventory

        inventory(cfg.data, cfg.inventory)
    elif cfg.action == "match":
        from .match import match

        match(cfg.data, cfg.inventory, cfg.selection, cfg.heldout)
    elif cfg.action == "catalog":
        from .adapters import catalog

        catalog(cfg.data, cfg.selection, cfg.catalog)
    elif cfg.action == "pack":
        pack(cfg.data, cfg.catalog, cfg.packed)
    elif cfg.action == "stage":
        if (cfg.data / "prep-audit.json").exists():
            bound_audit(cfg.data, cfg.packed / "index.json")
        stage(cfg.packed, cfg.local, cfg.splits)
    elif cfg.action == "audit":
        from .audit import audit

        result = audit(cfg.local)
        result["index_sha256"] = digest(cfg.local / "index.json")
        (cfg.data / "audit.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        if not result["pass"]:
            raise SystemExit("Nuisance controls failed")
    elif cfg.action == "pilot":
        from .pilot import pilot

        audit_report = bound_audit(cfg.data, cfg.local / "index.json")
        cfg.languages = sorted(set(cfg.languages))
        if audit_report.get("languages_audited", ["en", "hi"]) != cfg.languages:
            raise ValueError("Pilot languages differ from the audited languages")
        cfg.exception = audit_report.get("exception")
        pilot(cfg)
    elif cfg.action in {"teacher", "student"}:
        from .train import train

        if cfg.action == "student":
            from .exploratory import refuse

            refuse(cfg.data, "Student training")

        audit = bound_audit(cfg.data, cfg.local / "index.json")
        report = json.loads((cfg.data / "pilot.json").read_text())
        from .exploratory import pilot_authorizes

        if report.get("index_sha256") != audit["index_sha256"] or not (
            report.get("go") or pilot_authorizes(cfg.data, report, audit)
        ):
            raise ValueError("Pilot criteria have not passed for this pack")
        if report.get("languages", ["en", "hi"]) != audit.get("languages_audited", ["en", "hi"]):
            raise ValueError("Pilot and audit languages differ")
        cfg.exception = audit.get("exception")
        if report.get("exception") != cfg.exception:
            raise ValueError("Pilot exception differs from the bound audit")
        train(cfg, cfg.action == "student")
    elif cfg.action in {"export", "evaluate", "benchmark"} and (
        (cfg.data / "exception.json").exists() or cfg.data.name.endswith("-exploratory")
    ):
        raise ValueError(f"{cfg.action} is not allowed for an exploratory failed-audit run")
    elif cfg.action == "export":
        from .export import export

        export(cfg)
    elif cfg.action == "evaluate":
        from .export import evaluate

        evaluate(cfg)
    elif cfg.action == "benchmark":
        from .export import benchmark

        benchmark(cfg.release, cfg.threads)


if __name__ == "__main__":
    main()
