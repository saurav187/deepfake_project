import json

import pytest

from pipeline_v3 import exploratory as ex
from pipeline_v3 import teacher_eval as te
from pipeline_v3.data import digest
from pipeline_v3.stage0 import exploratory_ok, write_csv


def failed_source(data):
    src = data / ex.SOURCE
    src.mkdir(parents=True)
    write_csv(src / "selection.csv", [{"id": "a"}])
    write_csv(src / "eligible.csv", [{"id": "a", "sha256": "x"}])
    report = {
        "pass": False,
        "failed_fields": ["joint", "tilt"],
        "controls": {"joint": {"auc": 0.758}, "tilt": {"auc": 0.758}},
        "selection_sha256": digest(src / "selection.csv"),
        "eligible_sha256": digest(src / "eligible.csv"),
        "treatment": {"treatment": "norm"},
    }
    (src / "prep-audit.json").write_text(json.dumps(report))
    return report


def test_create_copies_the_failed_report_and_records_the_exception(tmp_path):
    report = failed_source(tmp_path)
    rec = ex.create(tmp_path)
    work = tmp_path / ex.WORK
    assert rec["failed_fields"] == report["failed_fields"]
    assert rec["failed_report_sha256"] == digest(tmp_path / ex.SOURCE / "prep-audit.json")
    assert digest(work / "prep-audit.json") == rec["failed_report_sha256"]
    assert ex.exception_for(work, report) == rec
    with pytest.raises(ValueError):
        ex.create(tmp_path)  # once only


def test_create_refuses_a_passing_source(tmp_path):
    failed_source(tmp_path)
    p = tmp_path / ex.SOURCE / "prep-audit.json"
    r = json.loads(p.read_text())
    r["pass"] = True
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        ex.create(tmp_path)


def test_failed_report_is_refused_outside_the_exploratory_dir(tmp_path):
    report = failed_source(tmp_path)
    with pytest.raises(ValueError):
        ex.exception_for(tmp_path / ex.SOURCE, report)
    assert not exploratory_ok(tmp_path / ex.SOURCE, report)  # default gate path: plain refusal


def test_exception_breaks_if_the_failed_report_changes(tmp_path):
    report = failed_source(tmp_path)
    ex.create(tmp_path)
    work = tmp_path / ex.WORK
    (work / "prep-audit.json").write_text(json.dumps(dict(report, failed_fields=["joint"])))
    with pytest.raises(ValueError):
        ex.exception_for(work, json.loads((work / "prep-audit.json").read_text()))


def test_passing_report_needs_no_exception(tmp_path):
    assert ex.exception_for(tmp_path, {"pass": True}) is None


def test_student_and_release_refuse_exploratory(tmp_path):
    failed_source(tmp_path)
    ex.create(tmp_path)
    with pytest.raises(ValueError):
        ex.refuse(tmp_path / ex.WORK, "Student training")
    ex.refuse(tmp_path / ex.SOURCE, "Student training")  # ordinary dir: allowed


def test_cli_refuses_export_for_exploratory(tmp_path, monkeypatch):
    import sys

    from pipeline_v3 import cli

    failed_source(tmp_path)
    ex.create(tmp_path)
    for action in ("student", "export", "evaluate", "benchmark"):
        monkeypatch.setattr(sys, "argv", ["cli", action, "--data", str(tmp_path / ex.WORK)])
        with pytest.raises((ValueError, FileNotFoundError)) as err:
            cli.main()
        if action != "student":
            assert "exploratory" in str(err.value)


def test_verdict_bands():
    assert te.verdict(0.9, 0.4, 0.6) == "likely_fake"
    assert te.verdict(0.5, 0.4, 0.6) == "uncertain"
    assert te.verdict(0.1, 0.4, 0.6) == "likely_real"


def test_summary_counts_clips_readers_voices_and_rates():
    items = [
        dict(label="0", generator="genuine", speaker="s1", voice="", channel="clean", verdict="likely_fake"),
        dict(label="0", generator="genuine", speaker="s2", voice="", channel="clean", verdict="likely_real"),
        dict(label="1", generator="xtts_v2", speaker="s1", voice="p254", channel="clean", verdict="likely_fake"),
        dict(label="1", generator="xtts_v2", speaker="s2", voice="p254", channel="clean", verdict="uncertain"),
    ]
    s = te.summarise(items, lambda x: (x["generator"], x["channel"]))
    assert s["genuine/clean"]["false_alarm_rate"] == 0.5
    assert s["genuine/clean"]["readers"] == 2
    assert s["xtts_v2/clean"]["detection_rate"] == 0.5
    assert s["xtts_v2/clean"]["voices"] == 1 and s["xtts_v2/clean"]["uncertain"] == 1


def test_channel_rng_is_deterministic():
    assert te._rng("a", "noise").integers(1 << 30) == te._rng("a", "noise").integers(1 << 30)


def test_panels_are_deterministic_and_outside_folds(tmp_path):
    (tmp_path / "stage0").mkdir()
    write_csv(tmp_path / "stage0" / "inventory.csv", [
        dict(group="dev-clean", uid=f"84-121-{i:04}", reader="84", chapter="121",
             archive="librispeech/dev-clean.tar.gz", member=f"m{i}", transcript="t")
        for i in range(30)
    ] + [dict(group="real", uid="1-2-0003", reader="1", chapter="2", archive="x", member="y", transcript="t")])
    (tmp_path / "stage0-en-ecapa-xtts").mkdir()
    write_csv(tmp_path / "stage0-en-ecapa-xtts" / "mlaad-diagnostic.csv", [
        dict(family="FishTTS", original="o", real_archive="mailabs/en_US.tgz",
             real_member="en_US/by_book/male/elliot_miller/b/wavs/a.wav",
             fake_archive=f"mlaad/fake/en/FishTTS/a{i}.wav", fake_member="-", voice="v")
        for i in range(3)
    ])
    rows = te.panel_rows(tmp_path, ["libri-dev", "mlaad"])
    libri = [r for r in rows if r["panel"].startswith("libri")]
    assert len(libri) == te.LIBRI_PER_READER and all(r["label"] == "0" for r in libri)
    mlaad = [r for r in rows if r["panel"] == "mlaad"]
    assert sum(r["label"] == "1" for r in mlaad) == 3
    assert sum(r["label"] == "0" for r in mlaad) == 1  # shared real parent kept once
    assert {r["speaker"] for r in mlaad} == {"mailabs/elliot_miller"}
    assert te.panel_rows(tmp_path, ["libri-dev", "mlaad"]) == rows
