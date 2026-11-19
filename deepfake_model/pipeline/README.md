# Deepfake Call Detection — Full Pipeline

## Project Overview

This codebase implements a real-time deepfake call detection system trained on the IndieFake dataset.
It is the practical realisation of the threshold instability findings from the companion paper.

**Core insight:** AUC is stable across evaluation conditions, but decision thresholds are not.
This system addresses that with a three-strategy stabilizer + temporal smoother.

---

## File Structure

```
01_prepare_dataset.py      Crawl IndieFake folders → X_features.npy / y_labels.npy / speaker_ids.npy
02_train_cnn_lstm.py       Speaker-independent nested CV training of CNN+LSTM
03_threshold_stabilizer.py Threshold stabilization + temporal smoothing module (importable)
04_realtime_inference.py   Live microphone / file / batch inference
```

---

## Setup

### Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install librosa soundfile numpy scipy scikit-learn matplotlib tqdm
pip install sounddevice    # only needed for live microphone mode
```

### Dataset structure expected

```
/home/saurav/A_mount/dataset/indiefake/
    Speaker-1/
        Bonafides/   ← real audio files (.wav / .flac / .mp3)
        Deepfakes/   ← synthetic audio files
    Speaker-2/
        ...
    Unspecified_Audio_Samples/
        ...
```

---

## Step-by-Step Usage

### Step 1 — Prepare features

```bash
python 01_prepare_dataset.py \
    --dataset_root /home/saurav/A_mount/dataset/indiefake \
    --output_dir   ./data \
    --workers      4
```

Outputs:
- `data/X_features.npy`   shape (N, 128, 94, 1)
- `data/y_labels.npy`     shape (N,)   0=real  1=fake
- `data/speaker_ids.npy`  shape (N,)   integer speaker index
- `data/speaker_map.json` speaker name → integer mapping

### Step 2 — Train CNN+LSTM

```bash
python 02_train_cnn_lstm.py \
    --data_dir   ./data \
    --output_dir ./outputs \
    --epochs     10 \
    --outer      5 \
    --inner      2
```

On subsequent runs, skip retraining:
```bash
python 02_train_cnn_lstm.py --data_dir ./data --output_dir ./outputs --load_cache
```

Outputs:
- `outputs/deepfake_cnnlstm.pth`          trained checkpoint
- `outputs/fold_records_cnnlstm.pkl`      cached fold results
- `outputs/fig1_roc.pdf`                  ROC curves
- `outputs/fig2_threshold_boxplot.pdf`    instability visualisation
- `outputs/fig3_strategy_comparison.pdf`  threshold strategy comparison
- `outputs/fig4_det.pdf`                  DET curves
- `outputs/fig5_calibration.pdf`          calibration ECE

### Step 3 — Test the threshold stabilizer (optional)

```bash
python 03_threshold_stabilizer.py
```

This runs a quick self-contained demo with simulated data.

### Step 4 — Real-time inference

**Test on a single audio file:**
```bash
python 04_realtime_inference.py \
    --checkpoint outputs/deepfake_cnnlstm.pth \
    --mode       file \
    --input_file path/to/test_audio.wav
```

**Live microphone (30 seconds):**
```bash
python 04_realtime_inference.py \
    --checkpoint outputs/deepfake_cnnlstm.pth \
    --mode       mic \
    --duration   30
```

**Batch evaluate a folder:**
```bash
python 04_realtime_inference.py \
    --checkpoint outputs/deepfake_cnnlstm.pth \
    --mode       folder \
    --input_dir  path/to/audio/folder \
    --output_csv results.csv
```

---

## Model Architecture

```
Input: (B, 1, 128, 94) log-Mel spectrogram chunk

CNN Backbone
├─ Block 1: Conv(1→32)   + BN + ReLU + MaxPool2d  →  (B, 32, 64, 47)
├─ Block 2: Conv(32→64)  + BN + ReLU + MaxPool2d  →  (B, 64, 32, 23)
└─ Block 3: Conv(64→128) + BN + ReLU + MaxPool2d  →  (B,128, 16, 11)

AdaptiveAvgPool2d(8,8)  →  (B, 128, 8, 8)
Reshape               →  (B, seq=8, features=1024)

Bidirectional LSTM (hidden=256, layers=2)
└─ concat last hidden (fwd + bwd)  →  (B, 512)

FC Head
├─ Linear(512→128) + ReLU + Dropout(0.4)
└─ Linear(128→1)  →  logit → sigmoid → score
```

Total parameters: ~3.7M

---

## Threshold Stabilization

Instead of a single threshold, we use majority vote across three strategies:

| Strategy         | Input      | Threshold type       |
|------------------|------------|----------------------|
| Youden's J       | raw score  | Youden-optimal (CV)  |
| Temperature      | logit / T  | Youden-optimal (CV)  |
| Majority vote    | all above  | ≥ 2/3 votes          |

This directly addresses the 8.49× instability ratio identified in the paper.

---

## Temporal Smoothing

Per-chunk scores are smoothed with an Exponential Moving Average (EMA):

```
EMA_t = α × score_t + (1 − α) × EMA_{t-1}
```

Default: α = 0.35, window = 5 chunks (≈ 5 seconds)

A call is flagged as FAKE if ≥ 40% of warmed-up chunks score above threshold.

---

## Key Differences from Baseline (Paper)

| Feature                        | Baseline CNN (paper) | This system              |
|-------------------------------|----------------------|--------------------------|
| Architecture                  | 2-block CNN          | 3-block CNN + BiLSTM     |
| Evaluation                    | Random fold          | Speaker-independent      |
| Threshold strategy            | Youden only          | 3-strategy majority vote |
| Temporal smoothing            | None                 | EMA (window=5)           |
| Real-time inference           | No                   | Yes (mic + file + batch) |
| Instability ratio reported    | Yes (8.49×)          | Yes (updated)            |

---

## Reproducing Paper Results

To reproduce the original 2-block CNN results from the paper, use the existing
`ThresholdStability_Combined.ipynb` notebook with `X_features.npy` and `y_labels.npy`
produced by Step 1 of this pipeline.

---

## Citation

If you use this code, please cite:

> Dahal, A.P., Kumar, S., Verma, A.K. (2024). On the Instability of Decision Thresholds
> in Deepfake Audio Detection. Cluster Innovation Centre, University of Delhi.
