import numpy as np
import matplotlib.pyplot as plt

X = np.load("X_indiefake.npy")
y = np.load("y_indiefake.npy")

real = X[np.where(y==0)[0][0]]
fake = X[np.where(y==1)[0][0]]

# Only MFCC part
real_mfcc = real[0:20]
fake_mfcc = fake[0:20]

# Normalize for visualization
real_mfcc = (real_mfcc - real_mfcc.mean()) / real_mfcc.std()
fake_mfcc = (fake_mfcc - fake_mfcc.mean()) / fake_mfcc.std()

fig, axes = plt.subplots(1,2, figsize=(12,5))

axes[0].imshow(real_mfcc, aspect='auto', origin='lower', cmap='coolwarm')
axes[0].set_title("Real MFCC (Normalized)")

axes[1].imshow(fake_mfcc, aspect='auto', origin='lower', cmap='coolwarm')
axes[1].set_title("Fake MFCC (Normalized)")

plt.tight_layout()
plt.show()
