"""V2-90 raw data download (CPU session). Resumable; writes one tar/parquet per unit to Drive.

MLAAD English folders (gated; HF token from $HF_TOKEN), LibriTTS-R, VCTK and GLOBE v2 parquet
shards. Each unit is fetched to local scratch, tarred (MLAAD) and copied to Drive with a sha256
line in SHA256SUMS. Rerunning skips units already listed.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

MLAAD_REV = "30c3dec763fa4803f11c2df1557d3b0828395366"
TARGET_RELATED = [
    "Chatterbox", "Chatterbox-Turbo", "Resemble.ai (April 12th, 2025)",
    "ElevenLabs-v3", "ElevenLabs-Turbo-v2.5", "ElevenLabs-v2-Multilingual",
    "FishTTS", "Fish-S2-Pro", "Openaudio-S1-Mini",
    "Gemini-3.1-Flash-TTS",
]
ON_DRIVE = ["Chatterbox", "ElevenLabs-v3", "FishTTS", "Gemini-3.1-Flash-TTS"]  # development/mlaad copy
DIVERSE_PER_FOLDER = 300  # en_US rows only (training never uses en_UK of non-target folders)
DIVERSE_N = 20
DIVERSE = [
    "Cartesia.ai (Sonic-3)", "DeepGram", "Edge-TTS", "OpenAI TTS-1 HD", "minimax_speech-2.6-hd",
    "minimax_speech-02-turbo", "Inworld-TTS-2", "Rime-Coda", "Smallest-Lightning-v3.1", "Kyutai-TTS",
    "Higgs-Audio-V2", "Higgs-Audio-V3", "Index-TTS-2.0", "Llasa-3B", "Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen3-TTS-CustomVoice", "VoxCPM2", "f5-tts", "Spark-TTS-0.5B", "sesame_csm", "kokoro",
    "orpheus-tts-0.1-finetune", "tts_models_multilingual_multi-dataset_xtts_v2", "GPT-SoVITS", "ZONOS2",
    "VibeVoice-Realtime-0.5B", "Voxtral", "FireRedTTS-2.0", "Step-Audio-EditX", "dots.tts-base",
    "OmniVoice", "ZipVoice", "NeuTTS-Air", "PocketTTS", "Kitten-TTS-Nano-0.2", "RVC", "suno_bark",
    "Echo-TTS", "GLM-TTS", "LongCat-AudioDiT", "Hume TADA-3B-ML", "Dramabox",
]
EXTRA_MLAAD = ["RVC"]  # voice conversion (training only, en_US rows)
EXTRA_GENUINE = {"ami": ("edinburghcstr/ami", ["ihm/test-*", "ihm/train-0000[0-5]-*", "sdm/test-*"])}
GENUINE = {
    "librittsr": ("mythicinfinity/libritts_r", ["data/dev.clean/*", "data/dev.other/*", "data/test.clean/*",
                                                 "data/test.other/*", "data/train.clean.100/*"]),
    "vctk": ("sanchit-gandhi/vctk", ["data/*"]),
    "globe": ("MushanW/GLOBE_V2", ["data/test-*", "data/train-0000*"]),
}


def sha(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    out = Path(sys.argv[1])
    only = sys.argv[2] if len(sys.argv) > 2 else "all"
    scratch = Path(f"/content/v290-scratch-{only}")
    out.mkdir(parents=True, exist_ok=True)
    sums = out / f"SHA256SUMS.{only}"
    done = {l.split("  ", 1)[1] for l in sums.read_text().splitlines()} if sums.exists() else set()

    def publish(local, rel):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        shutil.copyfile(local, tmp)
        h = sha(local)
        if sha(tmp) != h:
            raise ValueError(f"copy mismatch {rel}")
        tmp.rename(dst)
        with sums.open("a") as f:
            f.write(f"{h}  {rel}\n")
        done.add(rel)

    t0 = time.time()
    drive_mlaad = Path("/content/drive/MyDrive/audio-v3/development/mlaad/fake/en")

    def pack(src, folder, rel):
        tpath = scratch / "t.tar"
        scratch.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tpath, "w") as t:
            t.add(src, arcname=folder)
        publish(tpath, rel)
        tpath.unlink()
        print(f"{time.time() - t0:7.0f}s mlaad {folder}", flush=True)

    for folder in (ON_DRIVE if only not in ("genuine", "extra") else []):
        rel = f"mlaad/{folder}.tar"
        if rel not in done:
            pack(drive_mlaad / folder, folder, rel)
    for folder in [f for f in TARGET_RELATED if f not in ON_DRIVE and only not in ("genuine", "extra")]:
        rel = f"mlaad/{folder}.tar"
        if rel in done:
            continue
        d = scratch / "mlaad"
        snapshot_download("mueller91/MLAAD", repo_type="dataset", revision=MLAAD_REV, local_dir=d,
                          allow_patterns=[f"fake/en/{folder}/*"], max_workers=8)
        pack(d / "fake/en" / folder, folder, rel)
        shutil.rmtree(d / "fake/en" / folder)
    import csv as _csv
    import hashlib as _h
    div = EXTRA_MLAAD if only == "extra" else (DIVERSE[:DIVERSE_N] if only != "genuine" else [])
    for folder in div:
        rel = f"mlaad/{folder}.tar"
        if rel in done:
            continue
        d = scratch / "mlaad"
        m = hf_hub_download("mueller91/MLAAD", f"fake/en/{folder}/meta.csv", repo_type="dataset",
                            revision=MLAAD_REV, local_dir=d)
        rows = [r for r in _csv.DictReader(open(m), delimiter="|") if r["original_file"].startswith("en_US/")]
        rows.sort(key=lambda r: _h.sha256(r["path"].encode()).hexdigest())
        names = [r["path"].split("/")[-1] for r in rows[:DIVERSE_PER_FOLDER]]
        snapshot_download("mueller91/MLAAD", repo_type="dataset", revision=MLAAD_REV, local_dir=d,
                          allow_patterns=[f"fake/en/{folder}/{n}" for n in names], max_workers=8)
        keep = set(names) | {"meta.csv"}
        src = d / "fake/en" / folder
        for f in src.iterdir():
            if f.name not in keep:
                f.unlink()
        pack(src, folder, rel)
        shutil.rmtree(src)
    gen = EXTRA_GENUINE if only == "extra" else (GENUINE if only != "mlaad" else {})
    for name, (repo, pats) in gen.items():
        d = scratch / name
        snapshot_download(repo, repo_type="dataset", local_dir=d, allow_patterns=pats, max_workers=16)
        for p in sorted(d.rglob("*.parquet")):
            rel = f"{name}/{p.relative_to(d)}"
            if rel not in done:
                publish(p, rel)
            p.unlink()
        print(f"{time.time() - t0:7.0f}s genuine {name}", flush=True)
    print("DOWNLOAD COMPLETE", flush=True)


if __name__ == "__main__":
    main()
