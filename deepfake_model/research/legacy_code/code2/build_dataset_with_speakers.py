import os
import librosa
import numpy as np
import h5py
from tqdm import tqdm

DATASET_ROOT = "/home/saurav/A_mount/dataset/indiefake"

SR = 16000
WINDOW_SEC = 1.28
WINDOW = int(SR * WINDOW_SEC)

N_MFCC = 20
FRAME_TARGET = 128

h5file = h5py.File("indiefake_segments.h5", "w")

Xset = h5file.create_dataset(
    "X",
    shape=(0,60,128),
    maxshape=(None,60,128),
    dtype="float32",
    chunks=True
)

yset = h5file.create_dataset(
    "y",
    shape=(0,),
    maxshape=(None,),
    dtype="int"
)

count = 0

def extract_features(audio):

    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat = np.vstack([mfcc,delta,delta2])

    if feat.shape[1] < FRAME_TARGET:
        pad = FRAME_TARGET - feat.shape[1]
        feat = np.pad(feat,((0,0),(0,pad)))
    else:
        feat = feat[:,:FRAME_TARGET]

    return feat


for speaker in os.listdir(DATASET_ROOT):

    speaker_path = os.path.join(DATASET_ROOT,speaker)

    if not os.path.isdir(speaker_path):
        continue

    for label_name in ["Bonafides","Deepfakes"]:

        path = os.path.join(speaker_path,label_name)

        if not os.path.exists(path):
            continue

        label = 0 if label_name=="Bonafides" else 1

        for file in tqdm(os.listdir(path),desc=speaker):

            file_path = os.path.join(path,file)

            try:

                audio,_ = librosa.load(file_path,sr=SR)

                audio = audio / (np.max(np.abs(audio))+1e-9)

                start=0

                while start+WINDOW <= len(audio):

                    seg = audio[start:start+WINDOW]

                    feat = extract_features(seg)

                    Xset.resize((count+1,60,128))
                    yset.resize((count+1,))

                    Xset[count] = feat
                    yset[count] = label

                    count += 1

                    start += WINDOW//2

            except:
                pass

print("Total segments:",count)

h5file.close()