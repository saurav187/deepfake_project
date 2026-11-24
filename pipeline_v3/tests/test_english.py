import pytest

from pipeline_v3 import english
from pipeline_v3.data import check
from pipeline_v3.metrics import slices


def fixture(root, subset):
    for i in range(35):
        for j in range(25):
            uid = f"{subset[-1]}{i}-10-{j}"
            yield uid, f"a distinct sentence for person {uid}", subset, uid + ".flac"


def test_matched_english(monkeypatch, tmp_path):
    monkeypatch.setattr(english, "transcripts", fixture)
    jobs = english.plan(tmp_path)
    rows = english.combine([], jobs, False)
    assert len(jobs) == 1440
    for split, n in english.QUOTAS.items():
        fold = [r for r in jobs if r["split"] == split]
        assert len({r["speaker"] for r in fold}) == n * 2
        assert len({r["text"] for r in fold}) == n * 48
    assert all(r["engine"] != "gemini" for r in jobs if r["split"] == "train")
    final = english.plan(tmp_path, release=True)
    assert [r for r in final if r["split"] != "train"] == [
        r for r in jobs if r["split"] != "train"
    ]
    assert sum(r["engine"] == "gemini" and r["split"] == "train" for r in final) == 384
    assert not any(r["domain"] == "unseen" for r in english.combine([], final, True))
    rows[-1]["voice"] = rows[1]["voice"]
    with pytest.raises(ValueError, match="leaks"):
        check(rows)


def test_missing_speakers_stops(monkeypatch, tmp_path):
    monkeypatch.setattr(english, "transcripts", lambda *_: [])
    with pytest.raises(ValueError, match="eligible readers"):
        english.plan(tmp_path)


def test_unseen_metrics_require_own_language_parent():
    rows = [
        {
            "language": "en",
            "label": "1",
            "domain": "unseen",
            "generator": "gemini",
            "parent": "a",
            "pair": "book",
            "channel": "clean",
        },
        {
            "language": "en",
            "label": "0",
            "domain": "real",
            "generator": "-",
            "parent": "b",
            "pair": "book",
            "channel": "clean",
        },
    ]
    assert slices(rows, [1, 0])["language/en/unseen"]["eer"] is None
    rows[1]["parent"] = "a"
    assert slices(rows, [1, 0])["language/en/unseen"]["eer"] == 0


def test_archive_join_and_catalog_formats(tmp_path):
    import csv
    import io
    import tarfile
    import wave

    from pipeline_v3.adapters import catalog
    from pipeline_v3.generate import generated_path
    from pipeline_v3.match import write_selection

    root = tmp_path / "development"
    (root / "librispeech").mkdir(parents=True)
    (root / "generated").mkdir()
    audio = io.BytesIO()
    with wave.open(audio, "wb") as w:
        w.setparams((1, 2, 24000, 0, "NONE", ""))
        w.writeframes(b"\x02\x00" * 24000)
    audio = audio.getvalue()
    stem = "LibriSpeech/dev-clean/1/2/1-2"
    with tarfile.open(root / "librispeech/dev-clean.tar.gz", "w:gz") as tar:
        for name, content in [
            (
                stem + ".trans.txt",
                b"1-2-0001 A VALID TRANSCRIPT FOR THIS SOURCE RECORDING\n",
            ),
            (stem + "-0001.flac", audio),
        ]:
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            tar.addfile(entry, io.BytesIO(content))
    uid, text, archive, member = english.transcripts(root, "dev-clean")[0]
    job = {
        "id": uid,
        "text": text,
        "archive": archive,
        "member": member,
        "engine": "azure",
        "voice": "demo",
        "speaker": "reader",
        "split": "train",
    }
    (root / generated_path(job)).write_bytes(audio)
    write_selection(root / "joined.csv", english.combine([], [job], False))
    catalog(tmp_path, "joined.csv", "catalog.csv")
    with (root / "catalog.csv").open() as f:
        for row in csv.DictReader(f):
            with wave.open(str(root / row["path"])) as wav:
                assert (wav.getframerate(), wav.getsampwidth()) == (16000, 2)
