import numpy as np


def compress_array(arr):
    res_arr = np.zeros((28, 28))

    for i in range(0, len(arr), 10):
        for j in range(0, len(arr[0]), 10):
            s = 0

            for k in range(i, i + 10):
                for h in range(j, j + 10):
                    s += arr[k][h]

            s /= 100
            s *= 255
            res_arr[i // 10][j // 10] = int(s)

    return res_arr


def relu(x):
    return (x > 0) * x


def relu_derivative(x):
    return x > 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)
