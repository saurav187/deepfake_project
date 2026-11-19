import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

import h5py
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


DATA_PATH="indiefake_logmel.h5"

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE=64
EPOCHS=30
LR=1e-3


class AudioDataset(Dataset):

    def __init__(self,split):

        with h5py.File(DATA_PATH,"r") as f:

            X=f["X"][:]
            y=f["y"][:]
            s=f["split"][:]

        mask=s==split

        self.X=X[mask]
        self.y=y[mask]

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):

        x=torch.tensor(self.X[idx],dtype=torch.float32)
        y=torch.tensor(self.y[idx],dtype=torch.float32)

        return x,y


train_loader=DataLoader(AudioDataset(0),batch_size=BATCH_SIZE,shuffle=True)
val_loader=DataLoader(AudioDataset(1),batch_size=BATCH_SIZE)
test_loader=DataLoader(AudioDataset(2),batch_size=BATCH_SIZE)


class CNN_LSTM(nn.Module):

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


model=CNN_LSTM().to(DEVICE)

criterion=nn.BCEWithLogitsLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=LR)


best_val=0


for epoch in range(EPOCHS):

    model.train()

    loss_sum=0

    for x,y in tqdm(train_loader):

        x=x.to(DEVICE)
        y=y.to(DEVICE)

        optimizer.zero_grad()

        out=model(x)

        loss=criterion(out,y)

        loss.backward()

        optimizer.step()

        loss_sum+=loss.item()

    model.eval()

    preds=[]
    labels=[]

    with torch.no_grad():

        for x,y in val_loader:

            x=x.to(DEVICE)

            out=model(x)

            p=torch.sigmoid(out).cpu().numpy()

            preds.extend(p>0.5)
            labels.extend(y.numpy())

    acc=accuracy_score(labels,preds)

    print("Epoch",epoch+1,"Val acc:",acc)

    if acc>best_val:

        best_val=acc
        torch.save(model.state_dict(),"logmel_model.pt")


print("Best val:",best_val)