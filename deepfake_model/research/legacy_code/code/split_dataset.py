import os
import random
import pandas as pd

DATASET_PATH = "/home/saurav/A_mount/dataset/indiefake"

random.seed(42)

# -------------------
# find speakers
# -------------------

speakers = []

for folder in os.listdir(DATASET_PATH):

    if "Speaker" in folder:

        path = os.path.join(DATASET_PATH, folder)

        # handle nested folders
        subfolders = os.listdir(path)

        if len(subfolders) == 1 and "Speaker" in subfolders[0]:

            speakers.append(os.path.join(folder, subfolders[0]))

        else:

            speakers.append(folder)

speakers = sorted(speakers)

print("Total speakers:", len(speakers))

# -------------------
# split speakers
# -------------------

random.shuffle(speakers)

n = len(speakers)

train_speakers = speakers[:int(0.7*n)]
val_speakers = speakers[int(0.7*n):int(0.85*n)]
test_speakers = speakers[int(0.85*n):]

print("Train:", len(train_speakers))
print("Val:", len(val_speakers))
print("Test:", len(test_speakers))

# -------------------
# collect files
# -------------------

rows = []

def collect_files(speaker_list, split):

    for speaker in speaker_list:

        base_path = os.path.join(DATASET_PATH, speaker)

        bonafide_path = os.path.join(base_path, "Bonafides")
        deepfake_path = os.path.join(base_path, "Deepfakes")

        if os.path.exists(bonafide_path):

            for f in os.listdir(bonafide_path):

                if f.endswith(".wav"):

                    rows.append({
                        "file": os.path.join(bonafide_path, f),
                        "label": 0,
                        "speaker": speaker,
                        "split": split
                    })

        if os.path.exists(deepfake_path):

            for f in os.listdir(deepfake_path):

                if f.endswith(".wav"):

                    rows.append({
                        "file": os.path.join(deepfake_path, f),
                        "label": 1,
                        "speaker": speaker,
                        "split": split
                    })

collect_files(train_speakers, "train")
collect_files(val_speakers, "val")
collect_files(test_speakers, "test")

# -------------------
# save metadata
# -------------------

df = pd.DataFrame(rows)

df.to_csv("split.csv", index=False)

print("\nSaved split.csv")

print(df.groupby("split")["label"].value_counts())