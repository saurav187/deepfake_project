import numpy as np

# Load data
X = np.load("X_indiefake.npy")
y = np.load("y_indiefake.npy")

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

print("Data type of X:", X.dtype)
print("Data type of y:", y.dtype)

print("First sample of X:\n", X[0])
print("First label:", y[0])
