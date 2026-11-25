import json

from pipeline_v3 import v2l
from pipeline_v3.match import prompt
from pipeline_v3.stage0 import write_csv


def test_candidates_respect_v1_ownership_text_and_complete_parents(tmp_path):
    d = tmp_path
    (d / "stage0-en-ecapa-xtts").mkdir()
    write_csv(d / "stage0-en-ecapa-xtts" / "selection.csv", [
        dict(speaker="librispeech/1", split="train", label="0", text="other"),
        dict(speaker="librispeech/2", split="validate", label="0", text="x"),
        dict(speaker="librispeech/3", split="train", label="0", text="hello world"),
        dict(speaker="librispeech/9", split="select", label="0", text=prompt("HELLO WORLD")),
    ])
    man = []
    for uid, sp in (("1_10_0_0", "1"), ("2_10_0_0", "2"), ("3_10_0_0", "3")):
        for g in ("genuine", *v2l.FAMILIES, "wavenet"):
            man.append(dict(parent="librisevoc/" + uid, speaker="librispeech/" + sp, generator=g,
                            member=f"LibriSeVoc/{g}/{uid}.wav", label="0" if g == "genuine" else "1"))
    (d / "v2/librisevoc").mkdir(parents=True)
    write_csv(d / "v2/librisevoc/manifest.csv", man)
    work = d / "v2" / v2l.VERSION
    work.mkdir(parents=True)
    write_csv(work / "lineage.csv", [
        dict(source="librisevoc", uid="1_10_0_0", speaker="librispeech/1", transcript="A new line", evidence="e"),
        dict(source="librisevoc", uid="2_10_0_0", speaker="librispeech/2", transcript="Something", evidence="e"),
        dict(source="librisevoc", uid="3_10_0_0", speaker="librispeech/3", transcript="Hello, world!", evidence="e"),
    ])
    rows, excluded = v2l.candidates(d, work)
    parents = {r["parent"] for r in rows}
    assert parents == {"librisevoc/1_10_0_0"}  # 2: V1 validate reader; 3: text also in V1 select
    assert len(rows) == 5 and {r["generator"] for r in rows} == {"-", *v2l.FAMILIES}
    assert excluded["reader_owned_by_v1_validate"] == 1 and excluded["text_in_other_v1_fold"] == 1


def test_bottleneck_names_the_dominant_stage():
    b = v2l.bottleneck([(0.5, 0.1, 0.01, 0.1)] * 10, wall=2.0, workers=4)
    assert b["dominant"] == "zip_read" and b["measured_clips_per_s"] == 5.0


def test_text_key_matches_librispeech_style():
    assert v2l.text_key("Hello, world!") == prompt("HELLO WORLD")
    assert v2l.text_key("It\u2019s here.") == prompt("IT'S HERE")
