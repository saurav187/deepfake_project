import numpy as np
import os

files = os.listdir("lj_features")

print("Total feature files:", len(files))

sample = np.load("lj_features/" + files[0])

print("Shape:", sample.shape)
print("Mean:", np.mean(sample))
print("Std:", np.std(sample))
print("Min:", np.min(sample))
print("Max:", np.max(sample))
