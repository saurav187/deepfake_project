import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import h5py
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# ======================
# CONFIG
# ======================

DATA_PATH = "indiefake_logmel_fast.h5"

BATCH_SIZE = 32      # smaller batch
EPOCHS = 20
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# ======================
# DATASET (memory safe)
# ======================

class DeepfakeDataset(Dataset):

    def __init__(self, split):

        self.file = h5py.File(DATA_PATH, "r")

        self.X = self.file["X"]
        self.y = self.file["y"]
        self.s = self.file["split"]

        self.indices = np.where(self.s[:] == split)[0]

        print(f"Split {split} size:", len(self.indices))

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        i = self.indices[idx]

        x = torch.tensor(self.X[i], dtype=torch.float32)

        y = torch.tensor(self.y[i], dtype=torch.float32)

        return x, y


train_dataset = DeepfakeDataset(0)
val_dataset = DeepfakeDataset(1)
test_dataset = DeepfakeDataset(2)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2
)

# ======================
# FAST CNN MODEL
# ======================

class FastCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Linear(64,1)

    def forward(self,x):

        x = self.net(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x.squeeze()


model = FastCNN().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ======================
# TRAINING LOOP
# ======================

best_val = 0

for epoch in range(EPOCHS):

    print("\nEpoch", epoch+1)

    model.train()

    total_loss = 0

    for x,y in tqdm(train_loader):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out,y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    # validation

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for x,y in val_loader:

            x = x.to(DEVICE)

            out = model(x)

            p = torch.sigmoid(out).cpu().numpy()

            preds.extend(p > 0.5)

            labels.extend(y.numpy())

    acc = accuracy_score(labels,preds)

    print("Loss:", total_loss/len(train_loader))
    print("Val Accuracy:", acc)

    if acc > best_val:

        best_val = acc

        torch.save(model.state_dict(), "fast_deepfake_model.pt")

        print("Model saved")


print("\nBest validation accuracy:", best_val)