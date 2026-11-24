#!/usr/bin/env bash
# Usage: DOWNLOAD_PLAN.sh DATA [all|--list|--sizes|dataset-key ...]
set -euo pipefail
if [[ $# -lt 1 || -z $1 ]]; then echo "Usage: $0 DATA [--list|--sizes|dataset-key ...]" >&2; exit 2; fi
repo_python="$(dirname -- "${BASH_SOURCE[0]}")/../venv/bin/python"
if [[ -x $repo_python ]]; then
    python_bin=${PYTHON:-$repo_python}
else
    python_bin=${PYTHON:-python3}
fi
exec "$python_bin" - "$@" 3<&0 <<'PY'
import os
import fcntl
import sys
import shutil
import hashlib
import json
import subprocess
import zipfile
import gzip
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import quote, urlencode

root = Path(sys.argv[1]).expanduser().resolve()
args = sys.argv[2:]
models = ["ElevenLabs-v3", "Chatterbox", "FishTTS", "Gemini-3.1-Flash-TTS"]
hf = {
    "mlaad": ("mueller91/MLAAD", "30c3dec763fa4803f11c2df1557d3b0828395366",
              ["fake/en/" + model for model in models]),
    "kathbath": ("ai4bharat/Kathbath", "5b9e92849222026d9141acba4e8434fe816396bf", ["hindi"]),
    "indicsynth": ("vdivyasharma/IndicSynth", "c0a10386b723717aff682f757bd67f72983f269f", ["Hindi"]),
    "indicvoices": ("ai4bharat/IndicVoices", "c96f9088f138cf89d419da7e8e643e1f05c00a87", ["hindi"]),
    "shrutilipi": ("ai4bharat/Shrutilipi", "0df6350ebcbc4331344da06f7fc99eeebcf1f755", ["hindi"]),
    "deepfake-eval": ("nuriachandra/Deepfake-Eval-2024", "964763ccdbb659ca1cf40dcc8ba31476c3bb14a2", ["audio-data"]),
}
keys = ["mailabs", "mlaad", "kathbath", "indicsynth", "indicvoices", "shrutilipi", "speechfake", "deepfake-eval", "voxenes", "llamapartial"]
sealed = {"deepfake-eval", "voxenes", "llamapartial"}
if args == ["--list"]:
    print("Tier A:", *keys[:5], "\nTier B:", *keys[5:7], "\nSealed:", *keys[7:])
    sys.exit()
sizes_only = args == ["--sizes"]
selected = keys if not args or args == ["all"] or sizes_only else args
if set(selected) - set(keys):
    sys.exit("Unknown dataset key. Use --list.")
if not root.is_dir():
    sys.exit("DATA must be an existing directory on the destination disk.")
os.environ.setdefault("HF_XET_CACHE", str(root / ".xet"))
os.environ.setdefault("HF_HUB_CACHE", str(root / ".hub"))
if not sizes_only:
    lock = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("Another download plan is using this DATA directory.")
answers = os.fdopen(3)

def hf_plan(key):
    from huggingface_hub import HfApi, RepoFile
    repo, rev, folders = hf[key]
    api = HfApi()
    entries = []
    for folder in folders:
        files = sorted((f for f in api.list_repo_tree(repo, folder, recursive=True, repo_type="dataset", revision=rev)
                        if isinstance(f, RepoFile)), key=lambda f: f.path)
        if key in ("kathbath", "indicsynth", "indicvoices", "shrutilipi"):
            files = [f for f in files if Path(f.path).name.startswith("train-") and f.path.endswith(".parquet")]
            step = {"indicsynth": 8, "indicvoices": 20, "shrutilipi": 98}.get(key, 1)
            files = files[::step]
        if not files:
            raise ValueError(f"No files matched {repo}/{folder}")
        entries.extend(files)
    if key == "deepfake-eval":
        metadata = [f for f in api.list_repo_tree(repo, repo_type="dataset", revision=rev)
                    if isinstance(f, RepoFile) and f.path == "audio-metadata-publish.csv"]
        if len(metadata) != 1:
            raise ValueError("Missing audio metadata file")
        entries.extend(metadata)
    return [(f.path, f.size, None, "sha256:" + f.lfs.sha256 if f.lfs else None) for f in entries]

def remote_plan(key):
    if key == "mailabs":
        return [("en_US.tgz", 8016794960, "https://ics.tau-ceti.space/data/Training/stt_tts/en_US.tgz", None)]
    if key == "voxenes":
        return [("voxenes-v1.zip", 19846813791, "https://api.kaggle.com/v1/datasets/download/interspeech2712/voxenes-2026?datasetVersionNumber=1", None)]
    if key == "llamapartial":
        with urlopen("https://zenodo.org/api/records/14214149", timeout=30) as response:
            return [(f["key"], f["size"], "https://zenodo.org/records/14214149/files/" + quote(f["key"]),
                     f["checksum"]) for f in json.load(response)["files"]]
    base = "https://www.modelscope.cn/api/v1/datasets/inclusionAI/SPEECHFAKE/repo"
    revision = "e315a544246df765336bab9595ed552520444c4a"
    entries = []
    for folder in ("metadata", "MD", "Real", ""):
        with urlopen(base + "/tree?" + urlencode(dict(Root=folder, Revision=revision, PageSize=100)), timeout=30) as response:
            files = json.load(response)["Data"]["Files"]
        if not files:
            raise ValueError(f"Empty SpeechFake folder {folder}")
        for f in files:
            if folder == "Real" and f["Path"] != "Real/CommonVoice.zip":
                continue
            if f["Type"] == "blob" and (folder or f["Path"] in ("LICENSE.txt", "README.md")):
                entries.append((f["Path"], f["Size"], base + "?" + urlencode(dict(Revision=revision, FilePath=f["Path"])), "sha256:" + f["Sha256"]))
    return entries

def valid_archive(path, name, size, checksum):
    if path.stat().st_size != size:
        return False
    if checksum:
        algo, digest = checksum.split(":")
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, algo).hexdigest() == digest
    if name.endswith(".tgz"):
        with gzip.open(path) as stream:
            while stream.read(8 << 20):
                pass
        return True
    with zipfile.ZipFile(path) as archive:
        return bool(archive.infolist()) and archive.testzip() is None

def fetch(key):
    entries = hf_plan(key) if key in hf else remote_plan(key)
    total = sum(size for _, size, _, _ in entries)
    print(f"{key}: {len(entries)} files, {total/1e9:.4f} GB", flush=True)
    if sizes_only:
        return
    dst = root / ("sealed" if key in sealed else "development") / key
    print(f"Download to {dst}? [y/N] ", end="", flush=True)
    if answers.readline().strip().lower() not in ("y", "yes"):
        return
    dst.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dst)
    # Credit completed files and resumable partial archives.
    needed = 0
    for name, size, url, _ in entries:
        final = dst / name
        if final.is_file() and final.stat().st_size == size:
            continue
        part = final.with_name(final.name + ".part")
        partial = part.stat().st_size if url and part.is_file() else 0
        needed += max(0, size - partial)
    if needed and free.free - needed < free.total // 10:
        raise ValueError("Insufficient space to preserve 10% headroom.")
    for name, size, url, checksum in entries:
        if key in hf:
            from huggingface_hub import hf_hub_download
            repo, rev, _ = hf[key]
            downloaded = Path(hf_hub_download(repo, name, repo_type="dataset", revision=rev, local_dir=dst))
            if downloaded.stat().st_size != size or (checksum and not valid_archive(downloaded, name, size, checksum)):
                raise ValueError(f"Hub file failed integrity check in {dst}")
            continue
        final = dst / name
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            if valid_archive(final, name, size, checksum):
                continue
            raise ValueError(f"Existing file failed verification: {final}")
        part = final.with_name(final.name + ".part")
        if not part.exists() or part.stat().st_size != size:
            subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
                            "--output", str(part), url], check=True)
        if not valid_archive(part, name, size, checksum):
            raise ValueError(f"Verification failed: {part}")
        part.rename(final)
errors = []
for key in selected:
    try:
        fetch(key)
    except (OSError, EOFError, ValueError, RuntimeError, ImportError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        errors.append(key)
        print(f"{key} failed: {exc}", file=sys.stderr)
if errors:
    sys.exit("Rerun for: " + " ".join(errors))
PY
