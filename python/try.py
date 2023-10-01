import numpy as np

data = np.load('data/intrinsics.npz')
print(data['K1'])