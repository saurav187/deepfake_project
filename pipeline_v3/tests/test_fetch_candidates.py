import json
import socket
import subprocess
import sys

import pytest

from pipeline_v3.fetch_candidates import main


@pytest.mark.parametrize("flags", [[], ["--dry-run"]])
def test_candidate_plan_is_offline(flags, monkeypatch, tmp_path, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("Planner attempted external I/O")

    monkeypatch.setattr(socket, "socket", unexpected)
    monkeypatch.setattr(subprocess, "Popen", unexpected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan",
            "--data",
            str(tmp_path),
            "--sources",
            "librisevoc",
            "codecfake-sample",
            *flags,
        ],
    )
    main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] and plan["total_bytes"] == 45046253997 + 486611663
    assert plan["files"][1]["sha256"].startswith("0df27f3b")
    assert all("--continue-at -" in f["command"] for f in plan["files"])
    assert not list(tmp_path.iterdir())
