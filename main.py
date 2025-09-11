import numpy as np
from mnist import MNIST
import random

import matplotlib.pyplot as plt
import matplotlib

from sklearn import neighbors

import knn

import math
from tkinter import *
from tkinter.ttk import Scale

import draw

matplotlib.use('TkAgg')

# dataset = MNIST('C:/Users/nedob/Programming/Data Science/Datasets/MNIST_handwritten_numbers/archive')
dataset = MNIST('C:/Users/79386/Programming/Data_science/Datasets/mnist')
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
knn_mnist = neighbors.KNeighborsClassifier(n_neighbors=3, p=2)
knn_mnist.fit(train_images, train_labels)

# predictions = knn_mnist.predict(test_images)
# print(np.mean(predictions == test_labels))

window = Tk()
p = draw.Draw(window, knn_mnist)
window.mainloop()

# check_weights_3l()

# training_nn_3_layers()
