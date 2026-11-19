import numpy as np
import matplotlib.pyplot as plt

X = np.load("X_indiefake.npy")
y = np.load("y_indiefake.npy")

real_avg = X[y==0][:, 0:20].mean(axis=0)
fake_avg = X[y==1][:, 0:20].mean(axis=0)

real_avg = (real_avg - real_avg.mean()) / real_avg.std()
fake_avg = (fake_avg - fake_avg.mean()) / fake_avg.std()

fig, axes = plt.subplots(1,2, figsize=(12,5))

axes[0].imshow(real_avg, aspect='auto', origin='lower', cmap='coolwarm')
axes[0].set_title("Average Real MFCC")

axes[1].imshow(fake_avg, aspect='auto', origin='lower', cmap='coolwarm')
axes[1].set_title("Average Fake MFCC")

plt.tight_layout()
plt.show()
