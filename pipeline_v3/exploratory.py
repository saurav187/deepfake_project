"""Owner-approved exploratory exception to the nuisance gate (25 Nov 2026).

A separately named English Teacher V1 baseline is trained on the frozen-norm PhonemeDF
XTTS-only selection, DESPITE its failed nuisance audit. The default gate is unchanged:
only a work directory named WORK and that carries a valid
exception.json bound to the exact failed report may pass catalog, pack, stage, pilot and
teacher. Student, export, release and sealed evaluation always refuse such a run.

  python -m pipeline_v3.exploratory create --data $D
"""

import argparse
import json
import shutil
from pathlib import Path

from .data import digest

NAME = "exploratory-failed-audit-v1"
EXPLORATORY_SUFFIX = "-exploratory"
SOURCE = "stage0-en-ecapa-xtts-norm"
WORK = SOURCE + EXPLORATORY_SUFFIX
COPIED = ("selection.csv", "eligible.csv", "prep-audit.json")


def create(data):
    src, dst = data / SOURCE, data / WORK
    report = json.loads((src / "prep-audit.json").read_text())
    if report.get("pass"):
        raise ValueError("Source audit passed; no exception is needed")
    if (
        digest(src / "selection.csv") != report["selection_sha256"]
        or digest(src / "eligible.csv") != report["eligible_sha256"]
    ):
        raise ValueError("Failed source report no longer matches its selection")
    dst.mkdir(parents=True, exist_ok=True)
    if (dst / "exception.json").exists():
        raise ValueError("Exception already recorded; it is created once")
    for name in COPIED:
        shutil.copyfile(src / name, dst / name)
        if digest(dst / name) != digest(src / name):
            raise ValueError("Copy differs from source: " + name)
    record = {
        "exception": NAME,
        "approved_by": "owner",
        "approved_on": "2026-11-25",
        "scope": "Separately named exploratory English Teacher V1 baseline. Not a pass of "
        "the nuisance gate. Cannot authorize a student, export, release, final training or "
        "sealed-set comparison, and is not comparable with a gate-passing run.",
        "default_gate": "unchanged: every blocking probe AUC <= 0.60 in both directions",
        "failed_report": f"{SOURCE}/prep-audit.json",
        "failed_report_sha256": digest(src / "prep-audit.json"),
        "failed_fields": report["failed_fields"],
        "failed_auc": {
            k: report["controls"][k]["auc"] for k in report["failed_fields"]
        },
        "treatment": (report.get("treatment") or {}).get("treatment"),
        "held_out": ["chatterboxtts"],
        "pilot_gate": "retained unchanged; Teacher V1 only if the pilot passes",
        "evaluation": "teacher and threshold frozen on calibrate/threshold folds before "
        "any validate or diagnostic-panel scoring; sealed sets untouched",
        "known_shortcuts": "fakes are band-limited and darker (upper_band, rolloff, tilt) "
        "with higher speech ratio; see stage0-en-ecapa-xtts-norm/explain/explain.json",
    }
    tmp = dst / "exception.tmp"
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(dst / "exception.json")
    return record


def exception_for(work, report):
    """The valid exception for this work dir and failed report, or None. Raises if a
    failing report is presented anywhere else or with a mismatched record."""
    if report.get("pass"):
        return None
    path = Path(work) / "exception.json"
    if Path(work).name != WORK or not path.exists():
        raise ValueError("Failed nuisance audit; only a recorded exploratory exception may proceed")
    record = json.loads(path.read_text())
    prep = Path(work) / "prep-audit.json"
    if (
        record.get("exception") != NAME
        or record.get("failed_report_sha256") != digest(prep)
        or record["failed_report_sha256"] != digest(Path(work).parent / SOURCE / "prep-audit.json")
        or record.get("failed_report") != f"{SOURCE}/prep-audit.json"
        or report.get("pass") is not False
        or (report.get("treatment") or {}).get("treatment") != "norm"
        or record.get("failed_fields") != report.get("failed_fields")
    ):
        raise ValueError("Exploratory exception does not match this failed report")
    return record


def refuse(work, action):
    if Path(work).name.endswith(EXPLORATORY_SUFFIX) or (Path(work) / "exception.json").exists():
        raise ValueError(f"{action} is not allowed for an exploratory failed-audit run")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["create"])
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(create(a.data), indent=2))


if __name__ == "__main__":
    main()
