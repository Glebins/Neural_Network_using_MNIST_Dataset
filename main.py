import numpy as np
import torch
from mnist import MNIST
import random
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib

from sklearn import neighbors

import knn

import math
from tkinter import *
from tkinter.ttk import Scale

import draw

matplotlib.use('TkAgg')

dataset = MNIST('C:/Users/nedob/Programming/Data Science/Datasets/MNIST_handwritten_numbers/archive')
# dataset = MNIST('C:/Users/79386/Programming/Data_science/Datasets/mnist')

train_images, train_labels = dataset.load_training()
test_images, test_labels = dataset.load_testing()
train_images, train_labels, test_images, test_labels = \
    np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

train_images = train_images / 255.0
test_images = test_images / 255.0

alpha = 1

train_images = train_images[0:int(len(train_images) * alpha)]
train_labels = train_labels[0:int(len(train_labels) * alpha)]
test_images = test_images[0:int(len(test_images) * alpha)]
test_labels = test_labels[0:int(len(test_labels) * alpha)]

print(test_images.shape, train_images.shape)

# knn_mnist = knn.KNN(k=10)

# knn_mnist = neighbors.KNeighborsClassifier(n_neighbors=3, p=2)
# knn_mnist.fit(train_images, train_labels)

# predictions = knn_mnist.predict(test_images)
# print(np.mean(predictions == test_labels))

custom_model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Dropout(0.1),
    nn.Linear(96 * 3 * 3, 256), nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 10)
)

custom_model.load_state_dict(torch.load("./nn_weights_without_transforms/49.pth"))

window = Tk()
p = draw.Draw(window, custom_model)
window.mainloop()

# check_weights_3l()

# training_nn_3_layers()
