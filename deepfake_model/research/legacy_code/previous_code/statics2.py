import numpy as np
import os
from tqdm import tqdm

folder = "lj_features"
files = os.listdir(folder)

all_means = []
all_stds = []

for f in tqdm(files):
    data = np.load(os.path.join(folder, f))
    all_means.append(np.mean(data))
    all_stds.append(np.std(data))

print("Global Mean:", np.mean(all_means))
print("Global Std:", np.mean(all_stds))
