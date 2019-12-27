import numpy as np
import package
import package.optim as optim

from sin_regression import load_dataset


def test(test_file):
    X_test, Y_test = load_dataset(test_file)
    predicts = net.forward(X_test)
    acc, loss = loss_func(predicts, Y_test)
    print("test acc: %2.1f, test loss: %.2f" % (acc * 100, loss))


def predict(Xs):
    outputs = net.forward(Xs)
    predicts = np.argmax(outputs, axis=-1)
    predicts += 1
    np.savetxt('pred.txt', predicts, fmt='%d')


def train(net, loss_func, train_path, test_path, batch_size, optimizer, epoch_num=1):
    X, Y = load_dataset(train_path)
    data_size = X.shape[0]
    for e in range(epoch_num):
        print('epoch %d' % e)
        batch_num = data_size // batch_size + 1
        for b in range(batch_num):
            begin = b * batch_size
            end = (b + 1) * batch_size if b < batch_num - 1 else data_size
            x = X[begin: end]
            y = Y[begin: end]

            predicts = net.forward(x)
            batch_acc, batch_loss = loss_func(predicts, y)
            eta = loss_func.gradient()
            net.backward(eta)
            optimizer.update()
        pass
        test(test_path)


if __name__ == "__main__":
    layers = [
        {'type': 'batchnorm', 'shape': 784},
        {'type': 'linear', 'shape': (784, 1024)},
        {'type': 'batchnorm', 'shape': 1024},
        {'type': 'relu'},
        {'type': 'dropout', 'drop_rate': 0.1},
        {'type': 'linear', 'shape': (1024, 256)},
        {'type': 'batchnorm', 'shape': 256},
        {'type': 'relu'},
        {'type': 'dropout', 'drop_rate': 0.05},
        {'type': 'linear', 'shape': (256, 12)}
    ]
    loss_func = package.CrossEntropyLoss()
    net = package.Net(layers)
    lr = 0.01
    batch_size = 128
    optimizer = optim.Adam(net.parameters, lr)

    train_path = 'word_train.npz'
    test_path = 'word_test.npz'
    train(net, loss_func, train_path, test_path, batch_size, optimizer, epoch_num=10)
    print("*** final test ***")
    test(test_path)

    # # added for prediction task
    # task_path = 'word_task.npz'
    # task_file = np.load(task_path)
    # Xs = task_file['X']
    # predict(Xs)