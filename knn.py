import collections
import itertools

import numpy as np
from array_functions import *


def get_accuracy_test(predicts, test_labeled_data):
    return np.mean(predicts == test_labeled_data)


class KNN:

    def __init__(self, k=3):
        self.train_y = None
        self.train_x = None
        self.k = k

    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def compute_distance(self, test_x, type_d='l2'):
        assert self.train_x.shape[1] == test_x.shape[1], \
            f"Train and test arrays have different dimensions: " \
            f"{self.train_x.shape} for trains vs {test_x.shape} for tests"

        if type_d == 'l2':
            dists = np.sqrt(np.sum(np.square(test_x[:, None, :] - self.train_x), axis=2))
        elif type_d == 'l1':
            dists = np.sum(np.abs(test_x[:, None, :] - self.train_x), axis=2)
        else:
            raise ValueError("unknown distance type")

        return dists

    def predict(self, test_x):
        dists = self.compute_distance(test_x)
        num_test = test_x.shape[0]

        predictions = np.zeros(num_test, dtype='int')

        for i in range(num_test):
            distances = dict(zip(dists[i], self.train_y))
            distances = dict(sorted(distances.items()))
            distances_k = dict(itertools.islice(distances.items(), self.k))
            nearest_neighbors = list(distances_k.values())

            predictions[i] = collections.Counter(nearest_neighbors).most_common()[0][0]

        return predictions

    def guess_number(self, pixels):
        pixels = compress_array(pixels)
        pixels = pixels.T
        pixels /= 255
        pixels = pixels.reshape((1, 784))

        print(self.predict(pixels))
