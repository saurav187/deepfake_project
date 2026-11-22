# detector — deepfake speech detection, v2 pipeline

A fresh pipeline built from what the legacy CNN–LSTM (`../deepfake_model/`) taught us. Its research reports are the evidence for each rule below.

| Legacy finding | Rule here |
|---|---|
| File format predicted the label (0.95 AUC from sample rate and channels alone) | One canonical decode (`audio.py`) for training, evaluation and the app. The audit keeps only format strata containing both labels, and **fails** if format metadata still predicts the label. |
| Duplicates crossed speaker folders; test labels selected the model | Exact decoded-audio deduplication across all sources; test copies win. Speaker-group splits. Test sets are never used for any choice. |
| 8 kHz phone channels flipped most genuine audio to FAKE | Codec augmentation on both classes (training stage). |
| A single corpus did not transfer (ASVspoof AUC ≈ 0.6) | Several training corpora with modern generators; the primary metric is a corpus never used for training (In-the-Wild). |
| Near-silent openings collapsed into identical inputs | Edge-silence trimming and loudness normalisation; short clips are repeat-padded, never zero-padded. |

## Data

| Source | Role | Notes |
|---|---|---|
| ASVspoof 5 (2024) train, A01–A08 | train | Official partition. Downloaded tar by tar. |
| ASVspoof 5 dev, A09–A16 | dev | Attacks unseen in training. |
| IndieFake | train / dev | 80/20 split by speaker folder. Only format-balanced strata survive the audit. |
| In-the-Wild (2022) | **test** | Real and fake public-figure audio from the internet. |
| VA collection | **test** | 44 files; IndieFake training copies are removed by the audit. |

Downloads (resumable, checksum-verified): `bash detector/datasets/fetch.sh`. Live progress: `bash detector/datasets/progress.sh`.

## Steps (run from the repository root)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
venv/bin/python -m detector.data.build_manifest            # decode + describe every available file (incremental)
venv/bin/python -m detector.data.audit                     # dedupe, splits, format balance, leakage gate
venv/bin/python -m detector.eval.legacy_baseline --source in_the_wild --out detector/runs/baselines/legacy_itw
```

Outputs go to `detector/runs/` (git-ignored): per-source manifests, `audit/manifest.csv` (every file with its role or exclusion reason), `audit/report.json`, and baseline results.

## Model A

A frozen self-supervised encoder (`wavlm-base-plus`, English, 94M parameters, or `wav2vec2-xls-r-300m`, 128 languages, 300M parameters) with a learned softmax mix of all its layers, followed by attentive statistics pooling and a small classifier head (`model/ssl.py`). Only the head and layer weights are trained.

- **Input**: canonical decode → edge-silence trim → loudness normalisation → random 4 s windows (repeat-padded).
- **Augmentation** (`augment.py`, both classes alike): gain, white or pink noise at 10–40 dB SNR, and in 60% of windows a real codec round trip: MP3 32/64/128k, AAC 48/96k, Opus 16k (VoIP) and 32k, 8 kHz µ-law or PCM.
- **Sampling**: every (source, label) cell is drawn equally often, so a corpus's fake fraction cannot stand in for the label.
- **Selection**: after each epoch, dev is scored clean and through MP3 32k; the epoch with the lowest mean per-source EER is kept; early stopping after 4 epochs without improvement. Test corpora are never loaded.
- **Weights**: encoders live in `detector/pretrained/` (git-ignored, pinned revisions). Only the trainable state (a few MB) is saved per run.

```sh
venv/bin/python -m detector.train.train --encoder wav2vec2-xls-r-300m --seed 42 --out detector/runs/models/xlsr_s42
venv/bin/python -m detector.train.train --encoder wavlm-base-plus --smoke --out /tmp/smoke   # 1-minute pipeline test
```

`run_model_a.sh` waits for the ASVspoof 5 train and dev tars, rebuilds the manifest and audit, and trains one seed of each encoder. Log: `detector/runs/model_a.log`.

## Next

Pick the encoder on dev, train two more seeds, calibrate thresholds on dev, then evaluate once on In-the-Wild and VA. Then `detector/inference.py` implements the app's detector interface (`docs/PRD_deepfake_audio_app.md`, section 7).
