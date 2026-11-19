import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import h5py
import numpy as np
from sklearn.metrics import confusion_matrix


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MFCC_DATA = "/home/saurav/project/college/code3/indiefake_dataset.h5"
LOGMEL_DATA = "/home/saurav/project/college/code/indiefake_logmel.h5"

MFCC_MODEL = "/home/saurav/project/college/code3/best_model_v3.pt"
LOGMEL_MODEL = "/home/saurav/project/college/code/logmel_model.pt"

BATCH_SIZE = 64


# ==============================
# DATASETS
# ==============================

class MFCCDataset(Dataset):

    def __init__(self):

        with h5py.File(MFCC_DATA,"r") as f:

            X=f["X"][:]
            y=f["y"][:]
            s=f["split"][:]

        mask = s==2

        self.X = X[mask]
        self.y = y[mask]

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):

        x=torch.tensor(self.X[idx],dtype=torch.float32)
        y=torch.tensor(self.y[idx],dtype=torch.float32)

        return x,y


class LogMelDataset(Dataset):

    def __init__(self):

        with h5py.File(LOGMEL_DATA,"r") as f:

            X=f["X"][:]
            y=f["y"][:]
            s=f["split"][:]

        mask = s==2

        self.X = X[mask]
        self.y = y[mask]

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):

        x=torch.tensor(self.X[idx],dtype=torch.float32)
        y=torch.tensor(self.y[idx],dtype=torch.float32)

        return x,y


# ==============================
# MFCC MODEL (same as training)
# ==============================

class MFCC_Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(1,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(
            input_size=64*15,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(

            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64,1)
        )

    def forward(self,x):

        x=x.unsqueeze(1)

        x=self.cnn(x)

        B,C,H,W=x.shape

        x=x.permute(0,3,1,2)

        x=x.reshape(B,W,C*H)

        x,_=self.lstm(x)

        x=x[:,-1,:]

        x=self.fc(x)

        return x.squeeze()


# ==============================
# LOGMEL MODEL
# ==============================

class LogMel_Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn=nn.Sequential(

            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm=nn.LSTM(
            input_size=64*32,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.fc=nn.Sequential(

            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64,1)
        )

    def forward(self,x):

        x=x.unsqueeze(1)

        x=self.cnn(x)

        B,C,H,W=x.shape

        x=x.permute(0,3,1,2)

        x=x.reshape(B,W,C*H)

        x,_=self.lstm(x)

        x=x[:,-1,:]

        x=self.fc(x)

        return x.squeeze()


# ==============================
# EVALUATION
# ==============================

def evaluate(model,loader):

    model.eval()

    preds=[]
    labels=[]

    with torch.no_grad():

        for x,y in loader:

            x=x.to(DEVICE)

            out=model(x)

            p=torch.sigmoid(out).cpu().numpy()

            preds.extend(p>0.5)
            labels.extend(y.numpy())

    cm=confusion_matrix(labels,preds)

    return cm


# ==============================
# MFCC MODEL 3
# ==============================

mfcc_loader = DataLoader(MFCCDataset(),batch_size=BATCH_SIZE)

mfcc_model = MFCC_Model().to(DEVICE)

mfcc_model.load_state_dict(torch.load(MFCC_MODEL,map_location=DEVICE))

cm_mfcc = evaluate(mfcc_model,mfcc_loader)

print("\nMFCC_Model_3 Confusion Matrix")
print(cm_mfcc)


# ==============================
# LOGMEL MODEL
# ==============================

logmel_loader = DataLoader(LogMelDataset(),batch_size=BATCH_SIZE)

logmel_model = LogMel_Model().to(DEVICE)

logmel_model.load_state_dict(torch.load(LOGMEL_MODEL,map_location=DEVICE))

cm_logmel = evaluate(logmel_model,logmel_loader)

print("\nLogMel_Model Confusion Matrix")
print(cm_logmel)