import numpy as np
from PIL import Image

Xs = []

for x in range(1, 1801):
    imagePath = "./test/" + str(x) + ".bmp"
    retVect = np.empty(784)
    img = Image.open(imagePath).convert('RGB')
    for i in range(28):
        for j in range(28):
            r, g, b = img.getpixel((j, i))
            retVect[28 * i + j] = int(r / 255)
    Xs.append(retVect)

Xs = np.array(Xs)

per = np.random.permutation(Xs.shape[0])
X0 = Xs[per, :]

np.savez('word_task.npz', X=X0)
