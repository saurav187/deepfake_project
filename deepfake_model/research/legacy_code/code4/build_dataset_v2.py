import os
import librosa
import numpy as np
import h5py
from tqdm import tqdm

DATASET_PATH = "/home/saurav/A_mount/dataset/indiefake"
OUTPUT_FILE = "indiefake_logmel_fast.h5"

SR = 16000
SEGMENT = int(2 * SR)

N_MELS = 128
HOP = 160
FFT = 512


train_speakers = range(1,36)
val_speakers = range(36,43)
test_speakers = range(43,51)


def get_split(s):

    if s in train_speakers:
        return 0
    elif s in val_speakers:
        return 1
    else:
        return 2


def peak_normalize(x):

    return x / (np.max(np.abs(x)) + 1e-9)


def extract_features(segment):

    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=SR,
        n_fft=FFT,
        hop_length=HOP,
        n_mels=N_MELS
    )

    logmel = librosa.power_to_db(mel)

    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)

    feat = np.stack([logmel, delta, delta2])

    if feat.shape[2] < 128:
        pad = 128 - feat.shape[2]
        feat = np.pad(feat, ((0,0),(0,0),(0,pad)))

    feat = feat[:,:,:128]

    return feat.astype(np.float32)


# ==============================
# CREATE HDF5 FILE
# ==============================

f = h5py.File(OUTPUT_FILE, "w")

X_ds = f.create_dataset(
    "X",
    shape=(0,3,128,128),
    maxshape=(None,3,128,128),
    dtype="float32",
    chunks=(1,3,128,128),
    compression="gzip"
)

y_ds = f.create_dataset(
    "y",
    shape=(0,),
    maxshape=(None,),
    dtype="int8"
)

split_ds = f.create_dataset(
    "split",
    shape=(0,),
    maxshape=(None,),
    dtype="int8"
)

count = 0

# ==============================
# PROCESS DATA
# ==============================

for speaker in range(1,51):

    speaker_dir = os.path.join(DATASET_PATH, f"Speaker-{speaker}")

    real_dir = os.path.join(speaker_dir, "Bonafides")
    fake_dir = os.path.join(speaker_dir, "Deepfakes")

    s = get_split(speaker)

    for folder, label in [(real_dir,0),(fake_dir,1)]:

        if not os.path.exists(folder):
            continue

        files = os.listdir(folder)

        for file in tqdm(files, desc=f"Speaker {speaker}"):

            path = os.path.join(folder, file)

            try:

                audio, _ = librosa.load(path, sr=SR)

                audio = peak_normalize(audio)

                for start in range(0, len(audio) - SEGMENT, SEGMENT):

                    seg = audio[start:start+SEGMENT]

                    feat = extract_features(seg)

                    # resize datasets
                    X_ds.resize(count+1, axis=0)
                    y_ds.resize(count+1, axis=0)
                    split_ds.resize(count+1, axis=0)

                    X_ds[count] = feat
                    y_ds[count] = label
                    split_ds[count] = s

                    count += 1

            except:
                continue


print("Total samples:", count)


# ==============================
# NORMALIZATION
# ==============================

print("Computing normalization...")

train_mask = split_ds[:] == 0

mean = X_ds[train_mask].mean()
std = X_ds[train_mask].std()

for i in tqdm(range(count), desc="Normalizing"):

    X_ds[i] = (X_ds[i] - mean) / (std + 1e-6)


f.attrs["mean"] = mean
f.attrs["std"] = std

f.close()

print("Dataset saved:", OUTPUT_FILE)