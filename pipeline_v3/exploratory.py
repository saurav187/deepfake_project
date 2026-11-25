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


PILOT_ADVISORY = "pilot-advisory.json"


def record_pilot_advisory(work):
    """Owner approval (25 Nov 2026): in the exploratory run only, the pilot's PERFORMANCE
    result is advisory. The completed pilot report is kept as is; this record binds to it.
    Integrity checks (pack index, audit exception, languages) remain mandatory."""
    import json

    from .cli import bound_audit

    work = Path(work)
    if not work.name.endswith(EXPLORATORY_SUFFIX) or not (work / "exception.json").exists():
        raise ValueError("Advisory pilot is only for the recorded exploratory run")
    out = work / PILOT_ADVISORY
    if out.exists():
        raise ValueError("Pilot advisory already recorded")
    pilot = json.loads((work / "pilot.json").read_text())
    audit = json.loads((work / "audit.json").read_text())
    if (pilot.get("index_sha256") != audit.get("index_sha256")
            or pilot.get("exception") != audit.get("exception")
            or pilot.get("languages") != audit.get("languages_audited")):
        raise ValueError("Completed pilot does not match this pack, exception and languages")
    record = {
        "advisory": "exploratory-pilot-advisory-v1",
        "approved_by": "owner",
        "approved_on": "2026-11-25",
        "scope": "Exploratory V1 only. The pilot ran to completion and its pass/fail result "
        "is kept unchanged in pilot.json; a missed performance target does not stop Teacher "
        "V1. The audit and pilot are not rerun. Data-integrity and execution checks stay.",
        "pilot_go": pilot.get("go"),
        "pilot_sha256": digest(work / "pilot.json"),
        "index_sha256": audit["index_sha256"],
        "exception": audit["exception"]["exception"],
    }
    out.write_text(json.dumps(record, indent=2))
    return record


def pilot_authorizes(work, pilot, audit):
    """True if the pilot passed, or (exploratory only) a valid advisory record binds to this
    exact completed pilot report. Integrity mismatches always refuse."""
    import json

    work = Path(work)
    if (pilot.get("index_sha256") != audit.get("index_sha256")
            or pilot.get("exception") != audit.get("exception")
            or pilot.get("languages") != audit.get("languages_audited")):
        return False
    if pilot.get("go") is True:
        return True
    path = work / PILOT_ADVISORY
    if not audit.get("exception") or not work.name.endswith(EXPLORATORY_SUFFIX) or not path.exists():
        return False
    rec = json.loads(path.read_text())
    return (rec.get("advisory") == "exploratory-pilot-advisory-v1"
            and rec.get("pilot_sha256") == digest(work / "pilot.json")
            and rec.get("index_sha256") == audit["index_sha256"]
            and rec.get("pilot_go") == pilot.get("go"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["create", "pilot-advisory"])
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    if a.action == "create":
        print(json.dumps(create(a.data), indent=2))
    else:
        print(json.dumps(record_pilot_advisory(a.data / WORK), indent=2))


if __name__ == "__main__":
    main()
