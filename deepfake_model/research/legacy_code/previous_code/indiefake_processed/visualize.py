import numpy as np
import matplotlib.pyplot as plt

X = np.load("X_indiefake.npy")

plt.figure(figsize=(8, 4))
plt.imshow(X[0], aspect='auto', origin='lower')
plt.title("MFCC + Delta + Delta2 (Sample 0)")
plt.xlabel("Time Frames")
plt.ylabel("Feature Index")
plt.colorbar()
plt.show()