import numpy as np
from mnist import MNIST
import random

from array_functions import *

import math
from tkinter import *
from tkinter.ttk import Scale

import draw

file_save_weights = ["w_2l_1.txt", "w_2l_2.txt", "w_3l_1.txt", "w_3l_2.txt"]
chosen_number = 3

dataset = MNIST('C:/Users/79386/Programming/Data_science/Datasets/mnist')
train_images, train_labels = dataset.load_training()
test_images, test_labels = dataset.load_testing()
train_images, train_labels, test_images, test_labels =\
    np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

train_images = train_images / 255.0
test_images = test_images / 255.0


def guess_number(pixels):
    ''' # weights_right, iters = read_weights_from_file_2l(file_save_weights[chosen_number])
    k = 12

    weights_0_1, weights_1_2, iters = read_weights_from_file_3l(file_save_weights[2])
    pixels = compress_array(pixels).T
    pixels = np.reshape(pixels, 784)

    pixels = pixels / 255

    for i in range(28):
        for j in range(28):
            print(int(pixels[28 * i + j] * 255), end="\t")
        print()

    # predict = get_predict(pixels[0], weights_right)
    layer_1 = relu(np.dot(pixels, weights_0_1))
    predict = np.dot(layer_1, weights_1_2) '''

    # just_guess

    # draw.draw_infographic(predict)
    # print(predict, np.argmax(predict), sep="\n\n", end="\n")




window = Tk()
p = draw.Draw(window)
window.mainloop()

# check_weights_3l()

# training_nn_3_layers()
