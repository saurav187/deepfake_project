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


def test_copied_exception_in_another_suffix_dir_is_refused(tmp_path):
    import shutil

    report = failed_source(tmp_path)
    ex.create(tmp_path)
    other = tmp_path / "other-exploratory"
    shutil.copytree(tmp_path / ex.WORK, other)
    with pytest.raises(ValueError):
        ex.exception_for(other, report)


def test_freeze_rejects_stale_pilot_and_best(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from pipeline_v3 import cli

    cfg = SimpleNamespace(data=tmp_path, local=tmp_path, encoder=tmp_path)
    audit = {"index_sha256": "one", "exception": {"exception": ex.NAME}, "languages_audited": ["en"]}
    monkeypatch.setattr(cli, "bound_audit", lambda *_: audit)
    pilot = {"go": True, "languages": ["en"], "exception": audit["exception"], "index_sha256": "other"}
    (tmp_path / "pilot.json").write_text(json.dumps(pilot))
    with pytest.raises(ValueError, match="Passing pilot"):
        te.freeze(cfg)
    best = tmp_path / "best.pt"
    torch.save({"model": {}, "exception": None, "index_sha256": "one"}, best)
    with pytest.raises(ValueError, match="Teacher checkpoint"):
        te.load_teacher(cfg, best, "cpu", audit)


def test_mlaad_pair_eligibility_is_per_channel():
    rows = [
        {"panel": "mlaad", "label": label, "parent": parent, "channel": channel}
        for label, parent, channel in (("0", "a", "clean"), ("1", "a", "clean"),
                                       ("0", "a", "opus"), ("1", "b", "clean"))
    ]
    assert te.matched_panel_scores(rows) == rows[:2]


def test_freeze_then_evaluate_binds_checkpoint_and_index(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from pipeline_v3 import audio, cli, data

    cfg = SimpleNamespace(data=tmp_path, local=tmp_path, encoder=tmp_path, dev=None, panels=[])
    runs = tmp_path / "runs"
    runs.mkdir()
    with pytest.raises(FileNotFoundError):
        te.evaluate(cfg)
    index = tmp_path / "index.json"
    index.write_text("fixture")
    audit = {"exception": {"exception": ex.NAME}, "index_sha256": digest(index), "languages_audited": ["en"]}
    monkeypatch.setattr(cli, "bound_audit", lambda *_: audit)
    (tmp_path / "pilot.json").write_text(json.dumps(
        {"go": True, "index_sha256": digest(index), "languages": ["en"], "exception": audit["exception"]}
    ))
    best = runs / "teacher.best.pt"
    torch.save({"exception": audit["exception"], "index_sha256": digest(index),
                "encoder": str(cfg.encoder), "model": {}}, best)
    monkeypatch.setattr(te, "_model", lambda *_: torch.nn.Identity())
    monkeypatch.setattr(te, "mean_logit", lambda model, raw, device: raw)
    monkeypatch.setattr(data, "payload", lambda local, r: float(r["label"]))
    monkeypatch.setattr(audio, "decode", lambda b: b)
    monkeypatch.setattr(audio, "model_view", lambda raw, r: raw)
    monkeypatch.setattr(audio, "channel", lambda raw, *args: raw)
    seen = []

    def rows(local, splits):
        split, = splits
        seen.append(split)
        return [{"id": f"{split}-{i}", "parent": str(i // 2), "speaker": str(i // 2),
                 "label": str(i % 2), "generator": "tts" if i % 2 else "-"} for i in range(100)]

    monkeypatch.setattr(data, "entries", rows)
    frozen = te.freeze(cfg)
    assert seen == ["calibrate", "threshold"]
    assert frozen["exception"] == audit["exception"]
    with pytest.raises(ValueError, match="already frozen"):
        te.freeze(cfg)
    report = te.evaluate(cfg)
    assert seen[-1] == "validate" and report["exception"] == frozen["exception"]
    assert report["validate_by_generator_channel"]["tts/clean"]["clips"] == 50
    index.write_text("different pack")
    with pytest.raises(ValueError, match="changed since"):
        te.evaluate(cfg)
    index.write_text("fixture")
    best.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="changed since"):
        te.evaluate(cfg)


def _bound(tmp_path, go):
    failed_source(tmp_path)
    rec = ex.create(tmp_path)
    work = tmp_path / ex.WORK
    audit = {"index_sha256": "idx", "exception": rec, "languages_audited": ["en"]}
    (work / "audit.json").write_text(json.dumps(audit))
    pilot = {"go": go, "index_sha256": "idx", "exception": rec, "languages": ["en"]}
    (work / "pilot.json").write_text(json.dumps(pilot))
    return work, audit, pilot


def test_advisory_pilot_authorizes_only_with_bound_record(tmp_path):
    work, audit, pilot = _bound(tmp_path, False)
    assert not ex.pilot_authorizes(work, pilot, audit)
    rec = ex.record_pilot_advisory(work)
    assert rec["pilot_go"] is False
    assert ex.pilot_authorizes(work, pilot, audit)
    with pytest.raises(ValueError):
        ex.record_pilot_advisory(work)  # once
    # integrity still enforced
    assert not ex.pilot_authorizes(work, dict(pilot, index_sha256="other"), audit)
    # a changed pilot report breaks the binding
    (work / "pilot.json").write_text(json.dumps(dict(pilot, extra=1)))
    assert not ex.pilot_authorizes(work, pilot, audit)


def test_advisory_refused_outside_exploratory(tmp_path):
    failed_source(tmp_path)
    src = tmp_path / ex.SOURCE
    (src / "pilot.json").write_text("{}")
    with pytest.raises(ValueError):
        ex.record_pilot_advisory(src)
    audit = {"index_sha256": "i", "exception": None, "languages_audited": ["en"]}
    assert not ex.pilot_authorizes(src, {"go": False, "index_sha256": "i", "exception": None, "languages": ["en"]}, audit)


def test_failure_report_counts_false_alarms_and_misses():
    items = [
        dict(label="0", panel="validate", channel="opus", generator="genuine", speaker="a", verdict="likely_fake"),
        dict(label="0", panel="validate", channel="opus", generator="genuine", speaker="b", verdict="likely_real"),
        dict(label="1", panel="validate", channel="clean", generator="chatterboxtts", speaker="a", verdict="likely_real"),
        dict(label="1", panel="validate", channel="clean", generator="chatterboxtts", speaker="b", verdict="uncertain"),
    ]
    fa = te.failures(items, "0")["validate/opus"]
    assert fa["false_alarm_rate"] == 0.5 and fa["readers"] == 2
    miss = te.failures(items, "1")["validate/chatterboxtts/clean"]
    assert miss["missed_rate"] == 0.5 and miss["not_detected_rate"] == 1.0
