import zipfile

from pipeline_v3 import v2_librisevoc as v2
from pipeline_v3.stage0 import read, write_csv


def make(tmp_path):
    z = tmp_path / v2.ZIP
    z.parent.mkdir(parents=True)
    with zipfile.ZipFile(z, "w") as f:
        for uid in ("103_1241_000004_000002", "19_198_000001_000000", "7_1_000000_000000"):
            f.writestr(f"LibriSeVoc/gt/{uid}.wav", b"x")
            for voc in v2.VOCODERS:
                if uid.startswith("7_") and voc == "wavernn":
                    continue  # incomplete set is excluded
                f.writestr(f"LibriSeVoc/{voc}/{uid}_gen.wav", b"y")
    (tmp_path / "stage0-en-ecapa-xtts").mkdir()
    write_csv(tmp_path / "stage0-en-ecapa-xtts" / "selection.csv", [
        dict(id="a", split="train", label="0", speaker="librispeech/103", parent="librispeech/103-1241-0000"),
        dict(id="b", split="validate", label="0", speaker="librispeech/19", parent="librispeech/19-198-0003"),
    ])
    (tmp_path / "stage0").mkdir()
    write_csv(tmp_path / "stage0" / "inventory.csv", [
        dict(group="real", reader="103"), dict(group="real", reader="19"), dict(group="dev-clean", reader="84"),
    ])


def test_manifest_matches_six_vocoders_and_reports_overlap(tmp_path):
    make(tmp_path)
    o = v2.manifest(tmp_path)
    rows = read(tmp_path / v2.OUT / "manifest.csv")
    assert o["librisevoc_genuine_matched"] == 2 and len(rows) == 14
    assert o["unmatched"]["gt_without_all_six"] == 1
    assert o["reader_overlap_by_v1_fold"] == {"train": 1, "validate": 1}
    assert o["chapter_overlap_by_v1_fold"] == {"train": 1, "validate": 1}
    assert {r["parent"] for r in rows if r["generator"] == "wavenet"} == {r["parent"] for r in rows if r["label"] == "0"}


def test_sample_uses_only_v1_train_readers_and_keeps_parent_groups(tmp_path):
    make(tmp_path)
    v2.manifest(tmp_path)
    rows = read(tmp_path / v2.OUT / "manifest.csv")
    s = v2.sample(rows, {"librispeech/103"})
    assert {r["speaker"] for r in s} == {"librispeech/103"}
    assert len(s) == 7 and v2.sample(rows, {"librispeech/103"}) == s
