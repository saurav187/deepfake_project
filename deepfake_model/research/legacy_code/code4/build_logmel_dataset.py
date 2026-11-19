import os
import librosa
import numpy as np
import h5py
from tqdm import tqdm

DATASET_PATH = "/home/saurav/A_mount/dataset/indiefake"
OUTPUT_FILE = "indiefake_logmel.h5"

SR = 16000
SEGMENT_LENGTH = int(2 * SR)

N_MELS = 128
HOP_LENGTH = 160
N_FFT = 512


train_speakers = range(1,36)
val_speakers = range(36,43)
test_speakers = range(43,51)


def get_split(speaker):

    if speaker in train_speakers:
        return 0
    elif speaker in val_speakers:
        return 1
    else:
        return 2


def peak_normalize(audio):

    return audio / (np.max(np.abs(audio)) + 1e-9)


def extract_logmel(segment):

    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    logmel = librosa.power_to_db(mel)

    if logmel.shape[1] < 128:
        pad = 128 - logmel.shape[1]
        logmel = np.pad(logmel,((0,0),(0,pad)))

    logmel = logmel[:,:128]

    return logmel.astype(np.float32)


X=[]
y=[]
split=[]
speaker_ids=[]


for speaker in range(1,51):

    speaker_dir = os.path.join(DATASET_PATH,f"Speaker-{speaker}")

    real_dir = os.path.join(speaker_dir,"Bonafides")
    fake_dir = os.path.join(speaker_dir,"Deepfakes")

    s = get_split(speaker)

    for folder,label in [(real_dir,0),(fake_dir,1)]:

        if not os.path.exists(folder):
            continue

        files=os.listdir(folder)

        for file in tqdm(files,desc=f"Speaker {speaker}"):

            path=os.path.join(folder,file)

            try:

                audio,_ = librosa.load(path,sr=SR)

                audio = peak_normalize(audio)

                for start in range(0,len(audio)-SEGMENT_LENGTH,SEGMENT_LENGTH):

                    segment = audio[start:start+SEGMENT_LENGTH]

                    features = extract_logmel(segment)

                    X.append(features)
                    y.append(label)
                    split.append(s)
                    speaker_ids.append(speaker)

            except:
                continue


X=np.array(X)
y=np.array(y)
split=np.array(split)
speaker_ids=np.array(speaker_ids)

print("Dataset shape:",X.shape)


train_mask = split==0

mean = np.mean(X[train_mask])
std = np.std(X[train_mask])

X=(X-mean)/std


with h5py.File(OUTPUT_FILE,"w") as f:

    f.create_dataset("X",data=X,compression="gzip")
    f.create_dataset("y",data=y)
    f.create_dataset("split",data=split)
    f.create_dataset("speaker",data=speaker_ids)

    f.attrs["mean"]=mean
    f.attrs["std"]=std


print("Dataset saved:",OUTPUT_FILE)