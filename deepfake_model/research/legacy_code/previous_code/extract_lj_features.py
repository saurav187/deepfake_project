import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ===============================
# CONFIGURATION
# ===============================
DATASET_PATH = "/home/saurav/project/college/dataset/LJSpeech-1.1/wavs"
OUTPUT_PATH = "lj_features"
SAMPLE_RATE = 16000
N_MFCC = 20

Path(OUTPUT_PATH).mkdir(exist_ok=True)

def extract_mfcc_with_deltas(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Delta
    delta = librosa.feature.delta(mfcc)

    # Delta-Delta
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack together (shape: 60 x time)
    combined = np.vstack([mfcc, delta, delta2])

    return combined

def process_dataset():
    wav_files = list(Path(DATASET_PATH).glob("*.wav"))

    print(f"Total files: {len(wav_files)}")

    for wav_file in tqdm(wav_files):
        features = extract_mfcc_with_deltas(wav_file)

        save_path = Path(OUTPUT_PATH) / f"{wav_file.stem}.npy"
        np.save(save_path, features)

    print("Feature extraction complete!")

if __name__ == "__main__":
    process_dataset()
