import numpy as np
from mnist import MNIST
import random

import knn

import math
from tkinter import *
from tkinter.ttk import Scale

import draw

dataset = MNIST('C:/Users/79386/Programming/Data_science/Datasets/mnist')
train_images, train_labels = dataset.load_training()
test_images, test_labels = dataset.load_testing()
train_images, train_labels, test_images, test_labels =\
    np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

train_images = train_images / 255.0
test_images = test_images / 255.0


knn_mnist = knn.KNN(k=3)

window = Tk()
p = draw.Draw(window, knn_mnist)
window.mainloop()

# check_weights_3l()

# training_nn_3_layers()
