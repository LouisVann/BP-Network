import numpy as np
from PIL import Image

Xs = []
Ys = []


def onehot(x):
    return np.eye(12, dtype=int)[x - 1]


for x in range(1, 13):
    print(x)
    for y in range(1, 621):
        imagePath = "./train/" + str(x) + "/" + str(y) + ".bmp"
        retVect = np.empty(784)
        img = Image.open(imagePath).convert('RGB')
        for i in range(28):
            for j in range(28):
                r, g, b = img.getpixel((j, i))
                retVect[28 * i + j] = int(r / 255)
        Xs.append(retVect)
        Ys.append(onehot(x))

Xs = np.array(Xs)
Ys = np.array(Ys)

per = np.random.permutation(Xs.shape[0])
X0 = Xs[per, :]
Y0 = Ys[per, :]

length = X0.shape[0]
pos = int(length * 0.8)
X1 = X0[0 : pos, :]
Y1 = Y0[0 : pos, :]
X2 = X0[pos+1 : length, :]
Y2 = Y0[pos+1 : length, :]

np.savez('word_train.npz', X=X1, Y=Y1)
np.savez('word_test.npz', X=X2, Y=Y2)
