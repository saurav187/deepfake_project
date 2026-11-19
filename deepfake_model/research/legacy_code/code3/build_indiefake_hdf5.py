import os
import librosa
import numpy as np
import h5py
from tqdm import tqdm

DATASET_PATH = "/home/saurav/A_mount/dataset/indiefake"
OUTPUT_FILE = "indiefake_dataset.h5"

SR = 16000
SEGMENT_LENGTH = int(1.28 * SR)

N_MFCC = 20
HOP_LENGTH = 160
N_FFT = 512

train_speakers = range(1, 36)
val_speakers = range(36, 43)
test_speakers = range(43, 51)


def get_split(speaker_id):
    if speaker_id in train_speakers:
        return 0
    elif speaker_id in val_speakers:
        return 1
    else:
        return 2


def peak_normalize(audio):
    return audio / (np.max(np.abs(audio)) + 1e-9)


def extract_features(segment):

    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=SR,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])

    if features.shape[1] < 128:
        pad = 128 - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad)))

    features = features[:, :128]

    return features.astype(np.float32)


print("Scanning dataset...")

X = []
y = []
speaker_ids = []
splits = []

for speaker in range(1, 51):

    speaker_dir = os.path.join(DATASET_PATH, f"Speaker-{speaker}")

    real_dir = os.path.join(speaker_dir, "Bonafides")
    fake_dir = os.path.join(speaker_dir, "Deepfakes")

    split = get_split(speaker)

    for folder, label in [(real_dir,0),(fake_dir,1)]:

        if not os.path.exists(folder):
            continue

        files = os.listdir(folder)

        for file in tqdm(files, desc=f"Speaker {speaker}"):

            path = os.path.join(folder, file)

            try:

                audio, sr = librosa.load(path, sr=SR)

                audio = peak_normalize(audio)

                for start in range(0, len(audio)-SEGMENT_LENGTH, SEGMENT_LENGTH):

                    segment = audio[start:start+SEGMENT_LENGTH]

                    features = extract_features(segment)

                    X.append(features)
                    y.append(label)
                    speaker_ids.append(speaker)
                    splits.append(split)

            except:
                continue


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
speaker_ids = np.array(speaker_ids, dtype=np.int32)
splits = np.array(splits, dtype=np.int8)


print("Dataset size:", X.shape)


print("Computing normalization...")

train_mask = splits == 0

mean = np.mean(X[train_mask])
std = np.std(X[train_mask])

X = (X - mean) / std


print("Saving HDF5 dataset...")

with h5py.File(OUTPUT_FILE, "w") as f:

    f.create_dataset("X", data=X, compression="gzip")
    f.create_dataset("y", data=y)
    f.create_dataset("speaker_id", data=speaker_ids)
    f.create_dataset("split", data=splits)

    f.attrs["mean"] = mean
    f.attrs["std"] = std


print("Dataset saved successfully!")