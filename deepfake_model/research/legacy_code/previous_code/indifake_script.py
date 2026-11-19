import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# =====================================
# CONFIGURATION
# =====================================
DATASET_PATH = "/home/saurav/A_mount/dataset/indiefake"
OUTPUT_DIR = "indiefake_processed"
SAMPLE_RATE = 16000
N_MFCC = 20
TARGET_FRAMES = 128
# =====================================

Path(OUTPUT_DIR).mkdir(exist_ok=True)


def is_valid_audio(file_path):
    """
    Check if file is a valid readable audio file.
    """
    try:
        with sf.SoundFile(file_path):
            return True
    except Exception:
        return False


def extract_features(file_path):
    """
    Extract MFCC + Delta + Delta-Delta and fix length.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE) #Converts audio to mono
                                                    #Resamples to 16 kHz
                                                    #Returns:
                                                    #y → waveform (1D signal)
                                                    #sr → 16000
        #So every audio clip is normalized to same sampling rate.

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=512,
            hop_length=160,
            win_length=400
        )

        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.vstack([mfcc, delta, delta2])

        # Fix time dimension
        if features.shape[1] > TARGET_FRAMES:
            features = features[:, :TARGET_FRAMES]
        else:
            pad_width = TARGET_FRAMES - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_dataset():
    all_features = []
    all_labels = []

    valid_files = 0
    invalid_files = 0

    print("Starting feature extraction...\n")

    for speaker in tqdm(os.listdir(DATASET_PATH)):
        speaker_path = Path(DATASET_PATH) / speaker

        if not speaker_path.is_dir():
            continue

        for category in ["Bonafides", "Deepfakes"]:
            category_path = speaker_path / category

            if not category_path.exists():
                continue

            label = 0 if category == "Bonafides" else 1

            for wav_file in category_path.glob("*.wav"):

                if not is_valid_audio(wav_file):
                    invalid_files += 1
                    continue

                features = extract_features(wav_file)

                if features is None:
                    invalid_files += 1
                    continue

                all_features.append(features)
                all_labels.append(label)
                valid_files += 1

    print("\nExtraction finished.")
    print(f"Valid files processed: {valid_files}")
    print(f"Invalid / skipped files: {invalid_files}")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    print("\nFinal dataset shape:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    np.save(Path(OUTPUT_DIR) / "X_indiefake.npy", X)
    np.save(Path(OUTPUT_DIR) / "y_indiefake.npy", y)

    print("\nSaved processed dataset to:", OUTPUT_DIR)


if __name__ == "__main__":
    process_dataset()