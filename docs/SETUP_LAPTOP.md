# Running the deepfake audio app on another computer

This guide moves the finished app (detector release **v1**) from the training PC to a laptop. The laptop runs the **same model with the same accuracy**. A GPU is not needed: the laptop's processor gives the same verdicts, just a few seconds slower per file.

There are two parts:

| Part | What it contains | How it gets to the laptop |
|---|---|---|
| **Code** | the app, the detector code, docs | GitHub (`git clone`) |
| **Bundle**, `deepfake_bundle_v1.tar` (1.2 GB) | the XLS-R encoder, the 3 trained model files, the release settings, the built web page | you copy it yourself (USB drive, Google Drive, etc.) |

Git never stores model weights, so the bundle is the only way to get the trained model. **Keep a backup of it.** The encoder can be downloaded again, but the 3 trained heads exist only in this file and on the training PC.

---

## Part 1 — On the training PC (do this once)

The bundle has already been made at `~/deepfake_bundle_v1.tar`. To make it again, for example after building a new web page, run from the repository root:

```sh
npm --prefix app/frontend run build        # builds the web page into app/frontend/dist
bash detector/make_bundle.sh               # writes ~/deepfake_bundle_v1.tar
```

Copy `deepfake_bundle_v1.tar` to a USB drive (any format works, since the file is under 4 GB) or upload it to Google Drive.

It contains these 17 files, each stored at the place it must go in the repository:

```
BUNDLE_SHA256SUMS                                             checksums, used by the install check
detector/pretrained/wav2vec2-xls-r-300m/config.json           encoder (1.2 GB)
detector/pretrained/wav2vec2-xls-r-300m/preprocessor_config.json
detector/pretrained/wav2vec2-xls-r-300m/pytorch_model.bin
detector/runs/release/v1/release.json                         calibration and thresholds
detector/runs/release/v1/release.sha256
detector/runs/models/final_wav2vec2-xls-r-300m_s42/best.pt    trained head 1 (720 KB)
detector/runs/models/final_wav2vec2-xls-r-300m_s43/best.pt    trained head 2
detector/runs/models/final_wav2vec2-xls-r-300m_s44/best.pt    trained head 3
  (plus config.json and summary.json for each head)
app/frontend/dist/...                                         the built web page (no Node.js needed on the laptop)
```

You don't need to place these one by one: extracting the tar in the repository root puts every file in the right folder (Part 2, step 4).

---

## Part 2 — On the laptop

### What the laptop needs

- Windows 10/11, macOS or Linux
- 8 GB RAM or more, and about 4 GB of free disk space (about 1.2 GB for the bundle and 2 GB for Python packages)
- An internet connection for the installs (not for using the app)

### Step 1: install Python 3.14, Git and ffmpeg

**Windows** (PowerShell):
```powershell
winget install Python.Python.3.14
winget install Git.Git
winget install Gyan.FFmpeg
```
Then **close and reopen PowerShell** so the new commands are found. If you use the python.org installer instead, tick "Add python.exe to PATH".

**macOS** (with [Homebrew](https://brew.sh)):
```sh
brew install python@3.14 git ffmpeg
```

**Linux:** Manjaro/Arch: `sudo pacman -S python git ffmpeg`. Ubuntu/Debian: `sudo apt install python3 python3-venv git ffmpeg`.

Check:
```sh
python --version     # 3.12 or newer works; 3.14 matches the training PC (on macOS/Linux the command may be python3)
ffmpeg -version
```

### Step 2: get the code from GitHub

```sh
git clone -b detector-v2 https://github.com/saurav187/deepfake_project.git
cd deepfake_project
```
If the repository is private, Git asks you to sign in to GitHub (a browser window, or a personal access token as the password).

**Every command from here on runs inside the `deepfake_project` folder.**

### Step 3: create the Python environment and install packages

**Windows** (PowerShell):
```powershell
python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
venv\Scripts\python -m pip install -r detector/requirements.txt
```

**macOS / Linux:**
```sh
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # Linux; on macOS use: venv/bin/python -m pip install torch==2.15.0
venv/bin/python -m pip install -r detector/requirements.txt
```

The `whl/cpu` line installs the small CPU-only PyTorch (about 200 MB instead of several GB). If the laptop has an NVIDIA GPU and you want to use it, take the install command for torch 2.15.0 from <https://pytorch.org/get-started/locally/> instead. That's optional, since the CPU gives the same verdicts.

### Step 4: extract the bundle into the repository

Put `deepfake_bundle_v1.tar` anywhere on the laptop, then, **inside `deepfake_project`**:

```sh
tar -xf "path/to/deepfake_bundle_v1.tar"
```
For example, on Windows: `tar -xf "$HOME\Downloads\deepfake_bundle_v1.tar"`. `tar` is built into Windows 10 and 11.

Afterwards you should have `deepfake_project/detector/pretrained/`, `deepfake_project/detector/runs/` and `deepfake_project/app/frontend/dist/`. If they appear one folder too deep, you extracted somewhere else: delete them and extract again from inside `deepfake_project`.

### Step 5: check the installation

```sh
venv\Scripts\python -m detector.check_install      # Windows
venv/bin/python -m detector.check_install          # macOS / Linux
```

Every line should say `ok`. The check also loads the model and analyses a test tone, and it prints how long that took. You may see a long "LOAD REPORT" table with `UNEXPECTED` keys: that's normal, since parts of the encoder used only for pre-training aren't needed.

### Step 6: start the app

**Windows** (PowerShell):
```powershell
$env:DETECTOR_BACKEND = "detector"
venv\Scripts\python -m uvicorn app.backend.main:app --port 8000 --no-access-log
```

**macOS / Linux:**
```sh
DETECTOR_BACKEND=detector venv/bin/python -m uvicorn app.backend.main:app --port 8000 --no-access-log
```

Wait for `Application startup complete`, then open **<http://localhost:8000>** in a browser. Stop the app with **Ctrl+C** in that terminal.

**From now on, starting the app is only step 6:** open a terminal in `deepfake_project` and run the start command again. Steps 1–5 are one-time.

---

## Using it well (demo tips)

- **Upload at least 5–10 seconds of speech.** On the In-the-Wild test, the model caught 98% of fakes in clips with 4 s or more of speech, but only about 60% in shorter clips.
- Clips under about 1.5 s are refused with "not enough audio". That's deliberate.
- Accepted: `.wav .mp3 .m4a .aac .ogg .opus .flac .webm .mp4 .mov`, up to 50 MB and 10 minutes.
- The first analysis after starting is a little slower (warm-up).
- Uploaded audio is analysed in memory and never saved.
- Measured results to quote are in `detector/RESULTS.md`: In-the-Wild AUC 0.957, EER 11.3%, against 0.839 and 23.4% for the old model.
- Before the demo, start the app once at home, upload a couple of files, and check you get verdicts.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg` / `ffprobe` FAIL in the check | Install ffmpeg (step 1) and open a new terminal |
| `BUNDLE_SHA256SUMS not found` or `missing …` | The tar wasn't extracted in the `deepfake_project` folder (step 4) |
| `changed …` | The file was damaged while copying; copy the tar again |
| `ModuleNotFoundError` | Use the venv's Python (`venv\Scripts\python` or `venv/bin/python`), or rerun step 3 |
| The page shows "model unavailable" | Make sure `DETECTOR_BACKEND` is set to `detector` in the same terminal before starting; read the error printed at startup |
| `address already in use` | Another program uses port 8000: use `--port 8001` and open <http://localhost:8001> |
| `No module named detector` | Run the commands from inside the `deepfake_project` folder |
| Very slow analysis | Normal on older laptops: expect a few seconds to tens of seconds per file on a CPU |
| Lost the bundle | Make it again on the training PC (Part 1). Without the PC, the encoder can be downloaded again (`facebook/wav2vec2-xls-r-300m` on Hugging Face), but the 3 trained heads cannot, so keep a backup |

## Optional settings

| Variable | Default | Meaning |
|---|---|---|
| `DETECTOR_BACKEND` | `mock` | `detector` = the real model; `mock` = fake results for UI testing |
| `DEVICE` | `auto` | `cpu` or `cuda`; `auto` uses an NVIDIA GPU if PyTorch can see one |
| `DETECTOR_RELEASE` | `detector/runs/release/v1` | which model release to load |
| `MAX_UPLOAD_MB` | `50` | largest accepted file |
