import numpy as np
import pandas as pd
import librosa
import h5py
from tqdm import tqdm

# ======================
# CONFIG
# ======================

CSV_PATH = "split.csv"
OUTPUT = "indiefake_logmel_v8_fastest.h5"

SR = 16000
DURATION = 2

SEGMENT = SR * DURATION

N_MELS = 128
HOP = 160
FFT = 512

EPS = 1e-6

FRAMES_PER_SEGMENT = SEGMENT // HOP

BATCH_SIZE = 64   # write multiple samples at once

bad_files = []

# ======================
# AUDIO LOAD
# ======================

def load_audio(path):

    try:
        y, _ = librosa.load(path, sr=SR)

        if len(y) == 0:
            bad_files.append(path)
            return None

        y = y / (np.max(np.abs(y)) + EPS)

        return y

    except:
        bad_files.append(path)
        return None


# ======================
# FEATURE
# ======================

def compute_feature(y):

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=FFT,
        hop_length=HOP,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(mel)

    d1 = librosa.feature.delta(mel)
    d2 = librosa.feature.delta(mel, order=2)

    return np.stack([mel, d1, d2], axis=0).astype(np.float32)


# ======================
# LOAD CSV
# ======================

df = pd.read_csv(CSV_PATH)

print("Files:", len(df))

# ======================
# COUNT SEGMENTS
# ======================

total_segments = 0

for row in tqdm(df.itertuples(), total=len(df)):

    y = load_audio(row.file)

    if y is None:
        continue

    frames = len(y) // HOP

    segments = max(1, frames // FRAMES_PER_SEGMENT)

    total_segments += segments

print("\nTotal segments:", total_segments)

# ======================
# CREATE HDF5
# ======================

f = h5py.File(OUTPUT, "w")

X_ds = f.create_dataset(

    "X",

    shape=(total_segments, 3, N_MELS, FRAMES_PER_SEGMENT),

    dtype=np.float32,

    chunks=(BATCH_SIZE, 3, N_MELS, FRAMES_PER_SEGMENT),

    compression="lzf"   # much faster than gzip
)

y_ds = f.create_dataset(
    "y",
    shape=(total_segments,),
    dtype=np.int8
)

split_ds = f.create_dataset(
    "split",
    shape=(total_segments,),
    dtype="S10"
)

# ======================
# WRITE IN BATCHES
# ======================

buffer_X = []
buffer_y = []
buffer_split = []

idx = 0

print("\nExtracting (fast batch mode)...")

for row in tqdm(df.itertuples(), total=len(df)):

    y = load_audio(row.file)

    if y is None:
        continue

    feat = compute_feature(y)

    total_frames = feat.shape[-1]

    segments = max(1, total_frames // FRAMES_PER_SEGMENT)

    for i in range(segments):

        start = i * FRAMES_PER_SEGMENT
        end = start + FRAMES_PER_SEGMENT

        seg = feat[:, :, start:end]

        if seg.shape[-1] < FRAMES_PER_SEGMENT:

            pad = FRAMES_PER_SEGMENT - seg.shape[-1]

            seg = np.pad(
                seg,
                ((0,0),(0,0),(0,pad)),
                mode="wrap"
            )

        buffer_X.append(seg)
        buffer_y.append(row.label)
        buffer_split.append(str(row.split).encode())

        # write batch
        if len(buffer_X) == BATCH_SIZE:

            X_ds[idx:idx+BATCH_SIZE] = buffer_X
            y_ds[idx:idx+BATCH_SIZE] = buffer_y
            split_ds[idx:idx+BATCH_SIZE] = buffer_split

            idx += BATCH_SIZE

            buffer_X.clear()
            buffer_y.clear()
            buffer_split.clear()

# write remaining
if len(buffer_X) > 0:

    X_ds[idx:idx+len(buffer_X)] = buffer_X
    y_ds[idx:idx+len(buffer_X)] = buffer_y
    split_ds[idx:idx+len(buffer_X)] = buffer_split

f.close()

print("\nSaved:", OUTPUT)

print("Bad files:", len(bad_files))

with open("bad_audio.txt","w") as f:

    for b in bad_files:
        f.write(b+"\n")

print("Done")