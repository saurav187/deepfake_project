import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import h5py
from sklearn.metrics import confusion_matrix, accuracy_score

# =================================
# PATHS
# =================================

MODEL_PATH = "fast_deepfake_model.pt"

DATASET_PATH = "/home/saurav/A_mount/dataset/indiefake/Unspecified_Audio_Samples"

TRAIN_DATASET = "indiefake_logmel_fast.h5"

# =================================
# AUDIO PARAMETERS
# =================================

SR = 16000
SEGMENT = int(2 * SR)

N_FFT = 512
HOP = 160
N_MELS = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================================
# LOAD NORMALIZATION
# =================================

with h5py.File(TRAIN_DATASET, "r") as f:
    MEAN = f.attrs["mean"]
    STD = f.attrs["std"]

print("Loaded normalization")
print("Mean:", MEAN)
print("Std:", STD)

# =================================
# MODEL
# =================================

class FastCNN_GRU(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(3,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.gru = nn.GRU(
            input_size=64*32,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(

            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64,1)
        )

    def forward(self,x):

        x = self.cnn(x)

        B,C,H,W = x.shape

        x = x.permute(0,3,1,2)

        x = x.reshape(B,W,C*H)

        x,_ = self.gru(x)

        x = torch.mean(x, dim=1)

        x = self.fc(x)

        return x.squeeze()

# =================================
# LOAD MODEL
# =================================

model = FastCNN_GRU().to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.eval()

print("Model loaded")

# =================================
# PREPROCESSING
# =================================

def peak_normalize(x):

    return x / (np.max(np.abs(x)) + 1e-9)


def extract_features(segment):

    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=SR,
        n_fft=N_FFT,
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

    feat = feat[:, :, :128]

    feat = (feat - MEAN) / (STD + 1e-6)

    return feat.astype(np.float32)

# =================================
# PREDICT FILE
# =================================

def predict_file(filepath):

    audio,_ = librosa.load(filepath, sr=SR)

    audio = peak_normalize(audio)

    probs = []

    for start in range(0, len(audio) - SEGMENT + 1, SEGMENT):

        segment = audio[start:start+SEGMENT]

        feat = extract_features(segment)

        feat = torch.tensor(feat).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():

            prob = torch.sigmoid(model(feat)).item()

        probs.append(prob)

    if len(probs) == 0:
        return 0,0

    final_prob = np.mean(probs)

    THRESHOLD = 0.08
    pred = 1 if final_prob > THRESHOLD else 0

    return pred, final_prob

# =================================
# EVALUATION
# =================================

labels = []
preds = []

print("\nEvaluating dataset...\n")

for label, folder in [(0,"Bonafides"), (1,"Deepfakes")]:

    path = os.path.join(DATASET_PATH, folder)

    for file in sorted(os.listdir(path)):

        filepath = os.path.join(path, file)

        pred,prob = predict_file(filepath)

        preds.append(pred)
        labels.append(label)

        tag = "FAKE" if pred==1 else "REAL"

        print(f"{file:30} fake_prob={prob:.3f} → {tag} (true={label})")

# =================================
# RESULTS
# =================================

cm = confusion_matrix(labels, preds)

acc = accuracy_score(labels, preds)

print("\nConfusion Matrix")
print(cm)

print("\nAccuracy:", acc)