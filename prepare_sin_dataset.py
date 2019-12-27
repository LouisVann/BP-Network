import numpy as np
import matplotlib.pyplot as plt

N = 10000
xs = np.random.random((N, 1)) * 2 * np.pi - np.pi  # [-pi, pi]
ys = np.sin(xs)

print(xs.shape)
print(ys.shape)

plt.plot(xs, ys, 'r*')
plt.show()

np.savez("SinData_train.npz", X=xs, Y=ys)
