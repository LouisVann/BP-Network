import numpy as np
import package
import package.optim as optim
import matplotlib.pyplot as plt

# global variables as test-dataset
N = 1000
x_test = np.random.random((N, 1)) * 2 * np.pi - np.pi  # [-pi, pi]
y_test = np.sin(x_test)


def load_dataset(file, transform=False):
    file = np.load(file)
    X = file['X']
    Y = file['Y']
    if transform:
        X = X.reshape(len(X), -1)
    return X, Y


def train(net, loss_func, train_file, optimizer, batch_size=512, epoch_num=100):
    X, Y = load_dataset(train_file)
    data_size = len(X)
    for e in range(epoch_num):
        batch_num = data_size // batch_size + 1
        for b in range(batch_num):
            begin = b * batch_size
            end = (b + 1) * batch_size if b < batch_num - 1 else data_size
            x = X[begin : end]
            y = Y[begin : end]

            predicts = net.forward(x)
            batch_loss = loss_func(predicts, y)
            eta = loss_func.gradient()
            net.backward(eta)
            optimizer.update()
        pass
        predicts = net.forward(x_test)
        print(e, ' MSE ', np.sqrt(loss_func(predicts, y_test)))


if __name__ == "__main__":
    layers = [
        {'type': 'linear', 'shape': (1, 16)},
        {'type': 'relu'},
        {'type': 'linear', 'shape': (16, 32)},
        {'type': 'relu'},
        {'type': 'linear', 'shape': (32, 8)},
        {'type': 'relu'},
        {'type': 'linear', 'shape': (8, 1)}
    ]  # list of dict
    loss_func = package.MSELoss()
    model = package.Net(layers)
    lr = 0.01
    batch_size = 256
    optimizer = optim.Adam(model.parameters, lr)
    train_path = 'SinData_train.npz'

    train(model, loss_func, train_path, optimizer, batch_size, epoch_num=1000)
    predicts = model.forward(x_test)
    print('final test: MSE ', np.sqrt(loss_func(predicts, y_test)))

    plt.plot(x_test, predicts, 'r*')
    plt.show()
