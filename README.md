# Deepfake detection project

Goal: train and validate an audio deepfake detector, then build an app that continuously assesses voice audio during a call.

This workspace was assembled from the existing project on 2026-11-19. It contains independent copies of the main resources. The original project is preserved.

## Repository layout

```text
deepfake_project/
├── README.md
├── deepfake_model/    # Training, inference, research and local data
└── <app-folder>/      # Add your app source here when ready
```

This repository contains the model workspace and will also contain the application in a separate sibling folder. The app folder has not been created yet.

## Quick setup on another PC

Clone this repository, then copy the data separately from the existing `deepfake_model` folder using a drive or shared storage. **All destinations below are relative to `deepfake_model/` inside the cloned repository.** Keep the folder names unchanged; no old absolute paths are needed for the main pipeline.

| Copy from the existing workspace | Put in the clone | Needed for |
| --- | --- | --- |
| `datasets/indiefake/` (whole folder, about 8.1 GB) | `datasets/indiefake/` | Preparing features / retraining from raw audio |
| `datasets/va_dataset/` (whole folder, about 1.7 GB) | `datasets/va_dataset/` | External real/fake audio evaluation |
| `pipeline/data/` (all four files, about 880 MB) | `pipeline/data/` | Training with existing features: `X_features.npy`, `y_labels.npy`, `speaker_ids.npy`, `speaker_map.json` |
| `pipeline/outputs/deepfake_cnnlstm.pth` (about 17 MB) | `pipeline/outputs/deepfake_cnnlstm.pth` | Running the existing detector |

**Minimum for trying inference:** copy the checkpoint and provide an audio file. **Minimum for training from existing features:** copy `pipeline/data/`. Copy the raw IndieFake folder if you want to rebuild features. Create destination folders if they do not exist.

Optional: copy the rest of `pipeline/outputs/` for old plots/cached results, `research/legacy_models/` for earlier models, and `research/papers/` for reference PDFs. These are not required to run the main detector. Data, weights and PDFs are excluded from Git; cloning alone does not download them.

From the cloned repository, enter the model folder with `cd deepfake_model`. Create a fresh environment on each computer; do not copy `.venv`:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Use `python3` instead of `python` if that is your Python command. For microphone mode, also run `python -m pip install sounddevice` and configure the local audio device.

Try a file from inside `deepfake_model/`:

```bash
python pipeline/04_realtime_inference.py --mode file --input_file "path/to/test.wav"
```

Before retraining, read [the known issues and development plan](deepfake_model/docs/NEXT_STEPS.md). The existing model is an experimental baseline.

## Model workspace contents

Paths in this section are relative to `deepfake_model/`.

| Folder | Purpose |
| --- | --- |
| `pipeline/` | Main four-stage Python pipeline, prepared features, existing checkpoint, cached results, figures and inference logs |
| `datasets/indiefake/` | Raw IndieFake training corpus |
| `datasets/va_dataset/` | Existing external evaluation audio, organized into real and fake folders |
| `research/legacy_code/` | Earlier preprocessing/model experiments and threshold-stability notebooks; reference material with historical paths |
| `research/legacy_models/` | Earlier saved model weights |
| `research/papers/` | Existing reference PDFs |
| `research/notes.txt` | Original experiment notes |
| `docs/NEXT_STEPS.md` | Known issues and development sequence |
| `copy_manifest.json` | Source, destination and byte count for each copied file |

The saved final features have 19,146 samples: 8,172 real and 10,974 fake. Their shape is `(19146, 128, 94, 1)`. The 50 group IDs include an unspecified-speaker group.

## Environment

The old virtual environment was not copied because its interpreter link is broken. No new environment has been installed yet. The dependency list is an initial, unpinned list inferred from the main pipeline, not a tested lockfile.

From `deepfake_model/`, when ready to set up execution:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Microphone support additionally needs `sounddevice` and a working PortAudio/audio-device setup. Hardware-specific PyTorch setup can be chosen when the training environment is established.

## Existing pipeline

The copied scripts use paths relative to their own location for default inputs and outputs, so moving this entire workspace preserves those defaults. Explicit relative CLI arguments still resolve from your current working directory.

The existing checkpoint is `pipeline/outputs/deepfake_cnnlstm.pth`. After environment setup, an exploratory file prediction can be run from `deepfake_model/`:

```bash
python pipeline/04_realtime_inference.py \
  --checkpoint pipeline/outputs/deepfake_cnnlstm.pth \
  --mode file --input_file /path/to/audio.wav
```

Do not interpret its predictions or historical metrics as validated deployment performance. Review `docs/NEXT_STEPS.md` before retraining. The copy's path defaults are updated; model and evaluation logic remain unchanged.

Historical CSVs, notebooks and logs preserve their original paths and results for provenance. `pipeline/README.md` is also historical documentation and contains behavior descriptions that differ from the current scripts.

## Scope of this preparation

Copied the principal raw training/evaluation data, final feature arrays and results, earlier source experiments, reference papers and model weights. Excluded old virtual environments, duplicate ZIP archives, redundant intermediate feature datasets, notebook checkpoints, and the separate LJSpeech/WaveFake resources that the main IndieFake pipeline does not use. Those resources remain in the original project.

The application has not been built yet. The existing microphone mode is a local audio prototype; it does not implement capture of both sides of a phone or VoIP call. The app platform and permitted call-audio source must be established before implementing that integration.
