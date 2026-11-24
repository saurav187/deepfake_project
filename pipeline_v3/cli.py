"""Common command interface for Colab notebooks and terminal sessions."""

import argparse
import json
from pathlib import Path

from .data import SPLITS, digest, pack, stage


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
        stage(cfg.packed, cfg.local, cfg.splits)
    elif cfg.action == "audit":
        from .audit import audit

        result = audit(cfg.local)
        result["policy"] = "source-strict-v1"
        result["index_sha256"] = digest(cfg.local / "index.json")
        (cfg.data / "audit.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        if not result["pass"]:
            raise SystemExit("Nuisance controls failed")
    elif cfg.action == "pilot":
        from .pilot import pilot

        audit_report = json.loads((cfg.data / "audit.json").read_text())
        if (
            audit_report.get("policy") != "source-strict-v1"
            or not audit_report["pass"]
            or audit_report["index_sha256"] != digest(cfg.local / "index.json")
        ):
            raise ValueError("A passing audit for this pack is required")
        pilot(cfg)
    elif cfg.action in {"teacher", "student"}:
        from .train import train

        audit = json.loads((cfg.data / "audit.json").read_text())
        if (
            audit.get("policy") != "source-strict-v1"
            or not audit["pass"]
            or audit["index_sha256"] != digest(cfg.local / "index.json")
        ):
            raise ValueError("A passing audit for this pack is required")
        report = json.loads((cfg.data / "pilot.json").read_text())
        if not report.get("go") or report.get("index_sha256") != audit["index_sha256"]:
            raise ValueError("Pilot criteria have not passed for this pack")
        train(cfg, cfg.action == "student")
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
