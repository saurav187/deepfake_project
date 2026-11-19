import numpy as np
y = np.load("y_indiefake.npy")

print("Real:", np.sum(y==0))
print("Fake:", np.sum(y==1))fgdgdf