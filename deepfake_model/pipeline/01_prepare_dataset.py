"""
01_prepare_dataset.py
=====================
Crawls the IndieFake dataset directory structure:

    /path/to/indiefake/
        Speaker-1/
            Bonafides/   ← real audio  (label 0)
            Deepfakes/   ← fake audio  (label 1)
        Speaker-2/
            ...
        Unspecified_Audio_Samples/
            ...

Produces three .npy files used by all downstream scripts:
    X_features.npy     shape (N, 128, 94, 1)   log-Mel spectrograms
    y_labels.npy       shape (N,)               0=real  1=fake
    speaker_ids.npy    shape (N,)               integer speaker index

Usage
-----
    python 01_prepare_dataset.py --dataset_root /home/saurav/A_mount/dataset/indiefake
                                 --output_dir   ./data
                                 --sr           16000
                                 --n_mels       128
                                 --n_frames     94
                                 --workers      4

Requirements
------------
    pip install librosa soundfile numpy tqdm
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa


# ─── Feature extraction parameters (must match training) ─────────────────────
SR          = 16_000     # target sample rate
N_MELS      = 128        # mel filter banks
N_FRAMES    = 94         # time frames  (≈ 0.94 s of audio at hop=10 ms)
WIN_LENGTH  = int(0.025 * SR)   # 25 ms window  = 400 samples
HOP_LENGTH  = int(0.010 * SR)   # 10 ms hop     = 160 samples
F_MIN       = 0
F_MAX       = 8_000
DB_MIN      = -80.0
DB_MAX      = 0.0
AUDIO_EXTS  = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def extract_logmel(path: str, sr: int = SR) -> np.ndarray | None:
    """
    Load audio, compute log-Mel spectrogram, pad/crop to (N_MELS, N_FRAMES).
    Returns float32 array of shape (N_MELS, N_FRAMES) or None on error.
    """
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        if len(y) < WIN_LENGTH:
            return None                          # too short to be useful

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS,
            n_fft=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            fmin=F_MIN,
            fmax=F_MAX,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)   # dB scale
        log_mel = np.clip(log_mel, DB_MIN, DB_MAX)        # clamp to [-80, 0]

        # ── Pad or crop time axis to exactly N_FRAMES ────────────────────
        T = log_mel.shape[1]
        if T < N_FRAMES:
            pad = N_FRAMES - T
            log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant",
                             constant_values=DB_MIN)
        else:
            log_mel = log_mel[:, :N_FRAMES]

        return log_mel.astype(np.float32)
    except Exception as e:
        print(f"  [WARN] skipped {path}: {e}")
        return None


def _worker(args):
    path, label, spk_id = args
    feat = extract_logmel(path)
    if feat is None:
        return None
    return feat, label, spk_id


def collect_file_list(dataset_root: str):
    """
    Walk dataset_root and collect (filepath, label, speaker_id) tuples.

    Directory rules
    ---------------
    - Any folder named 'Bonafides'  (case-insensitive) → label 0 (real)
    - Any folder named 'Deepfakes'  (case-insensitive) → label 1 (fake)
    - Speaker index is derived from the parent folder name.
    - Unspecified_Audio_Samples is treated as a separate speaker.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    file_list = []
    speaker_name_to_id = {}
    current_id = 0

    # Sort for reproducibility
    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir():
            continue

        spk_name = speaker_dir.name
        if spk_name not in speaker_name_to_id:
            speaker_name_to_id[spk_name] = current_id
            current_id += 1
        spk_id = speaker_name_to_id[spk_name]

        for class_dir in sorted(speaker_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            dname = class_dir.name.lower()
            if "bonafide" in dname or "real" in dname or "genuine" in dname:
                label = 0
            elif "deepfake" in dname or "fake" in dname or "spoof" in dname or "synth" in dname:
                label = 1
            else:
                print(f"  [INFO] skipping unrecognised folder: {class_dir}")
                continue

            for audio_file in sorted(class_dir.rglob("*")):
                if audio_file.suffix.lower() in AUDIO_EXTS:
                    file_list.append((str(audio_file), label, spk_id))

    print(f"\nSpeakers found  : {len(speaker_name_to_id)}")
    print(f"Files found     : {len(file_list)}")
    real_n = sum(1 for _, l, _ in file_list if l == 0)
    fake_n = sum(1 for _, l, _ in file_list if l == 1)
    print(f"  Real (label 0): {real_n}")
    print(f"  Fake (label 1): {fake_n}")
    print(f"  Imbalance ratio (fake/real): {fake_n/max(real_n,1):.3f}")
    return file_list, speaker_name_to_id


def build_arrays(file_list, n_workers: int = 4):
    """Extract features in parallel and stack into numpy arrays."""
    X, y, spk = [], [], []

    print(f"\nExtracting features with {n_workers} workers …")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, args): args for args in file_list}
        for fut in tqdm(as_completed(futures), total=len(file_list), unit="file"):
            result = fut.result()
            if result is not None:
                feat, label, spk_id = result
                X.append(feat)
                y.append(label)
                spk.append(spk_id)

    # Stack  →  (N, N_MELS, N_FRAMES)  then  add channel dim  →  (N, N_MELS, N_FRAMES, 1)
    X_arr   = np.stack(X, axis=0)[..., np.newaxis]   # (N,128,94,1)
    y_arr   = np.array(y,   dtype=np.int64)
    spk_arr = np.array(spk, dtype=np.int64)

    return X_arr, y_arr, spk_arr


def main():
    parser = argparse.ArgumentParser(description="Prepare IndieFake log-Mel features")
    parser.add_argument("--dataset_root", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "datasets" / "indiefake"),
                        help="Root directory of the IndieFake dataset")
    parser.add_argument("--output_dir",   type=str, default=str(Path(__file__).resolve().parent / "data"),
                        help="Where to save X_features.npy / y_labels.npy / speaker_ids.npy")
    parser.add_argument("--sr",      type=int, default=SR)
    parser.add_argument("--n_mels",  type=int, default=N_MELS)
    parser.add_argument("--n_frames",type=int, default=N_FRAMES)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Override globals if user changed them
    globals()["N_MELS"] = args.n_mels
    globals()["N_FRAMES"] = args.n_frames
    globals()["SR"] = args.sr

    print("=" * 60)
    print("IndieFake Dataset Preparation")
    print("=" * 60)
    print(f"Dataset root : {args.dataset_root}")
    print(f"Output dir   : {out}")
    print(f"SR={SR}  n_mels={N_MELS}  n_frames={N_FRAMES}")
    print(f"Feature shape per sample: (1, {N_MELS}, {N_FRAMES})")

    file_list, speaker_map = collect_file_list(args.dataset_root)
    X, y, spk = build_arrays(file_list, n_workers=args.workers)

    x_path   = out / "X_features.npy"
    y_path   = out / "y_labels.npy"
    spk_path = out / "speaker_ids.npy"

    np.save(x_path,   X)
    np.save(y_path,   y)
    np.save(spk_path, spk)

    # Save speaker name→id mapping
    import json
    with open(out / "speaker_map.json", "w") as f:
        json.dump(speaker_map, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  X_features.npy  : {X.shape}   dtype={X.dtype}")
    print(f"  y_labels.npy    : {y.shape}   Real={( y==0).sum()}  Fake={(y==1).sum()}")
    print(f"  speaker_ids.npy : {spk.shape}  unique speakers={len(np.unique(spk))}")
    print(f"  speaker_map.json: {len(speaker_map)} entries")
    print(f"  Feature range   : [{X.min():.1f}, {X.max():.1f}]  "
          f"μ={X.mean():.3f}  σ={X.std():.3f}")
    print("=" * 60)
    print(f"\nNext step: python 02_train_cnn_lstm.py --data_dir {out}")


if __name__ == "__main__":
    main()
