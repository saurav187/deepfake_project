import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import h5py
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

DATA_PATH = "indiefake_logmel_fast.h5"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
ACCUM_STEPS = 4
EPOCHS = 30
LR = 1e-3


# ==============================
# DATASET (Lazy Loading)
# ==============================

class AudioDataset(Dataset):

    def __init__(self, split):

        self.file = h5py.File(DATA_PATH, "r")

        X = self.file["X"]
        y = self.file["y"]
        s = self.file["split"]

        self.indices = np.where(s[:] == split)[0]

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        i = self.indices[idx]

        x = torch.tensor(self.X[i], dtype=torch.float32)
        y = torch.tensor(self.y[i], dtype=torch.float32)

        return x, y


train_loader = DataLoader(
    AudioDataset(0),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    AudioDataset(1),
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True
)


# ==============================
# SpecAugment
# ==============================

def spec_augment(x):

    if np.random.rand() < 0.5:

        fmask = np.random.randint(5,15)
        tmask = np.random.randint(10,30)

        f0 = np.random.randint(0,128-fmask)
        t0 = np.random.randint(0,128-tmask)

        x[:,f0:f0+fmask,:] = 0
        x[:,:,t0:t0+tmask] = 0

    return x


# ==============================
# MODEL
# ==============================

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

        x = x[:,-1,:]

        x = self.fc(x)

        return x.squeeze()


model = FastCNN_GRU().to(DEVICE)


# ==============================
# LOSS / OPTIMIZER
# ==============================

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    patience=3
)

scaler = GradScaler()

best_val = 0


# ==============================
# TRAIN LOOP
# ==============================

for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()

    for i,(x,y) in enumerate(tqdm(train_loader)):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        x = spec_augment(x)

        with autocast():

            out = model(x)

            loss = criterion(out,y) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (i+1) % ACCUM_STEPS == 0:

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()


    # =========================
    # VALIDATION
    # =========================

    model.eval()

    preds=[]
    labels=[]

    with torch.no_grad():

        for x,y in val_loader:

            x = x.to(DEVICE)

            with autocast():
                out = model(x)

            p = torch.sigmoid(out).cpu().numpy()

            preds.extend(p>0.5)
            labels.extend(y.numpy())

    acc = accuracy_score(labels,preds)

    scheduler.step(acc)

    print(f"Epoch {epoch+1} Val Acc: {acc:.4f}")

    if acc > best_val:

        best_val = acc

        torch.save(model.state_dict(),"fast_deepfake_model.pt")

        print("Model saved")


print("Best Validation Accuracy:",best_val)