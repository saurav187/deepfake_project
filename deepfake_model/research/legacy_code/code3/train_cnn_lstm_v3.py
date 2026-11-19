import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import h5py
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix


DATA_PATH = "indiefake_dataset.h5"

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


# ==============================
# DATASET
# ==============================

class DeepfakeDataset(Dataset):

    def __init__(self, hdf5_path, split):

        print(f"Loading split {split} into RAM...")

        with h5py.File(hdf5_path, "r") as f:

            X = f["X"][:]
            y = f["y"][:]
            s = f["split"][:]

        mask = s == split

        self.X = X[mask]
        self.y = y[mask]

        print("Loaded samples:", len(self.X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return x, y


# ==============================
# MODEL
# ==============================

class CNN_LSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(
            input_size=64*15,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.cnn(x)

        B,C,H,W = x.shape

        x = x.permute(0,3,1,2)

        x = x.reshape(B,W,C*H)

        x,_ = self.lstm(x)

        x = x[:,-1,:]

        x = self.fc(x)

        return x.squeeze()


# ==============================
# LOAD DATASETS
# ==============================

train_dataset = DeepfakeDataset(DATA_PATH, 0)
val_dataset = DeepfakeDataset(DATA_PATH, 1)
test_dataset = DeepfakeDataset(DATA_PATH, 2)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4
)


# ==============================
# MODEL
# ==============================

model = CNN_LSTM().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=LR)


# ==============================
# TRAINING LOOP
# ==============================

best_val = 0
patience = 5
counter = 0

for epoch in range(EPOCHS):

    print("\nEpoch",epoch+1)

    model.train()

    train_loss = 0

    for x,y in tqdm(train_loader):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(output,y)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # =========================
    # VALIDATION
    # =========================

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for x,y in val_loader:

            x = x.to(DEVICE)

            output = model(x)

            p = torch.sigmoid(output).cpu().numpy()

            preds.extend(p>0.5)
            labels.extend(y.numpy())

    val_acc = accuracy_score(labels,preds)

    print("Train Loss:",train_loss)
    print("Val Accuracy:",val_acc)


    # =========================
    # EARLY STOPPING
    # =========================

    if val_acc > best_val:

        best_val = val_acc
        counter = 0

        torch.save(model.state_dict(),"best_model_v3.pt")

        print("Model saved")

    else:

        counter += 1

        if counter >= patience:

            print("Early stopping triggered")
            break


# ==============================
# TEST EVALUATION
# ==============================

print("\nLoading best model...")

model.load_state_dict(torch.load("best_model_v3.pt"))

model.eval()

preds = []
labels = []

with torch.no_grad():

    for x,y in test_loader:

        x = x.to(DEVICE)

        output = model(x)

        p = torch.sigmoid(output).cpu().numpy()

        preds.extend(p>0.5)
        labels.extend(y.numpy())


acc = accuracy_score(labels,preds)

cm = confusion_matrix(labels,preds)

print("\nTest Accuracy:",acc)
print("\nConfusion Matrix:")
print(cm)