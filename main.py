import numpy as np
from mnist import MNIST
import random

from neural_network_functions import *

import math
from tkinter import *
from tkinter.ttk import Scale

file_save_weights = ["w_2l_1.txt", "w_2l_2.txt", "w_3l_1.txt", "w_3l_2.txt"]
chosen_number = 3

dataset = MNIST('C:/Users/nedob/Programming/Data Science/Datasets/MNIST_handwritten_numbers/archive')
images, labels = dataset.load_training()
images, labels = np.array(images), np.array(labels)

images = images / 255


class Draw:
    def __init__(self, root):

        self.infographic = None
        self.result = None
        self.canvas_width = 280
        self.canvas_height = 280

        # Defining title and Size of the Tkinter Window GUI
        self.pixels = np.zeros((self.canvas_width, self.canvas_height))
        self.root = root
        root.geometry("1000x600+550+200")
        self.root.title("Painter")
        self.root.configure(background="white")
        #         self.root.resizable(0,0)

        # variables for pointer and Eraser
        self.pointer = "black"

        # Reset Button to clear the entire screen
        self.clear_screen = Button(self.root, text="Clear Screen", bd=4, bg='white',
                                   command=self.clear_canvas, width=9, relief=RIDGE)
        self.clear_screen.place(x=0, y=30)

        self.print_image = Button(self.root, text="Print an array", bd=4, bg='white',
                                  command=lambda: self.print_array_of_pixels(), width=9, relief=RIDGE)
        self.print_image.place(x=0, y=70)

        self.guess = Button(self.root, text="Guess the number", bd=4, bg='white',
                            command=lambda: self.guess_number(), width=10, relief=RIDGE)
        self.guess.place(x=0, y=110)

        # Creating a Scale for pointer and eraser size
        self.pointer_frame = LabelFrame(self.root, text='size', bd=5, bg='white', font=('arial', 15, 'bold'),
                                        relief=RIDGE)
        self.pointer_frame.place(x=0, y=320, height=200, width=70)

        self.pointer_size = Scale(self.pointer_frame, orient=VERTICAL, from_=50, to=0, length=168)
        self.pointer_size.set(10)
        self.pointer_size.grid(row=0, column=1, padx=15)

        # Defining a background color for the Canvas
        # self.background = Canvas(self.root, bg='white', bd=5, relief=GROOVE, height=470, width=680)
        self.background = Canvas(self.root, bg='white', bd=5, relief=GROOVE, height=self.canvas_height,
                                 width=self.canvas_width)
        self.background.place(x=80, y=40)

        # Bind the background Canvas with mouse click
        self.background.bind("<B1-Motion>", self.paint)

    def clear_canvas(self):
        self.background.delete('all')
        self.pixels = np.zeros((self.canvas_width, self.canvas_height))

    def paint(self, event):
        radius = self.pointer_size.get()
        x1, y1 = (event.x - radius), (event.y - radius)
        x2, y2 = (event.x + radius), (event.y + radius)

        self.background.create_oval(x1, y1, x2, y2, fill=self.pointer, outline=self.pointer)

        for i in range(int(event.x - radius) + 1, int(event.x + radius) + 1):
            dy = math.sqrt(radius ** 2 - (event.x - i) ** 2)

            for j in range(int(event.y - dy), int(event.y + dy + 1)):
                if i < 0 or i > self.canvas_width - 1 or j < 0 or j > self.canvas_height - 1:
                    continue
                self.pixels[i, j] = 1

    def print_array_of_pixels(self):
        pixels = compress_array(self.pixels)

        for i in range(len(pixels)):
            for j in range(len((pixels[0]))):
                print(int(pixels[j][i]), end='\t')
            print()
        print("\n\n")

    def draw_infographic(self, predict):
        self.infographic = Canvas(self.root, width=550, height=320)
        self.infographic.place(x=400, y=40)

        for i in range(1, 11):
            self.infographic.create_text(20, 30 * i, text=str(i - 1), fill="black", font="Helvetica 15 bold")

        max_len = 300
        alpha = max_len / np.max(predict)

        for i in range(0, 10):
            self.infographic.create_line(50, 30 * (i + 1), alpha * predict[i] + 50, 30 * (i + 1), fill="green", width=0)

        self.result = Canvas(self.root, bg='white', bd=5, relief=GROOVE, height=40, width=40)
        self.result.place(x=500, y=400)
        self.result.create_text(30, 30, text=str(np.argmax(predict)), fill="black", font='Helvetica 15 bold')

    def guess_number(self):
        # weights_right, iters = read_weights_from_file_2l(file_save_weights[chosen_number])
        test_images, test_labels = dataset.load_testing()
        test_images, test_labels = np.array(test_images), np.array(test_labels)
        test_images = test_images / 255
        k = 12

        weights_0_1, weights_1_2, iters = read_weights_from_file_3l(file_save_weights[2])
        pixels = compress_array(self.pixels).T
        pixels = np.reshape(pixels, 784)

        pixels = pixels / 255

        for i in range(28):
            for j in range(28):
                print(int(pixels[28 * i + j] * 255), end="\t")
            print()

        # predict = get_predict(pixels[0], weights_right)
        layer_1 = relu(np.dot(pixels, weights_0_1))
        predict = np.dot(layer_1, weights_1_2)
        self.draw_infographic(predict)
        print(predict, np.argmax(predict), sep="\n\n", end="\n")


def test_nn_2_layers(weights):
    test_images, test_labels = dataset.load_testing()
    test_images, test_labels = np.array(test_images), np.array(test_labels)
    test_right_answers = get_goal_predictions_in_right_form(test_labels)

    test_predictions = np.dot(test_images, weights.T)
    test_error = get_accuracy_of_prediction(test_predictions, test_right_answers)

    goal_predict = get_goal_predictions_in_right_form(labels)

    prediction_on_training_data = np.dot(images, weights.T)
    error_training_data = get_accuracy_of_prediction(prediction_on_training_data, goal_predict)

    return test_error, error_training_data


def test_nn_3_layers(weights_0_1, weights_1_2):
    test_images, test_labels = dataset.load_testing()
    test_images, test_labels = np.array(test_images), np.array(test_labels)
    test_images = test_images / 255

    test_right_answers = get_goal_predictions_in_right_form(test_labels)

    layer_1_test = relu(np.dot(test_images, weights_0_1))
    test_predictions = np.dot(layer_1_test, weights_1_2)
    test_error = get_accuracy_of_prediction(test_predictions, test_right_answers)

    train_right_answers = get_goal_predictions_in_right_form(labels)

    layer_1_train = relu(np.dot(images, weights_0_1))
    train_prediction = np.dot(layer_1_train, weights_1_2)
    train_error = get_accuracy_of_prediction(train_prediction, train_right_answers)

    return test_error, train_error


def get_predict(inputs, weights):
    predict = np.dot(inputs, weights.T)
    return predict


def print_random_image_from_dataset():
    index = random.randrange(0, len(images))

    for i in range(28):
        for j in range(28):
            print(images[index][28 * i + j], end="\t")

        print()

    print("\n\n", labels[index])


def get_goal_predictions_in_right_form(labels):
    res = np.zeros((len(labels), 10))

    for i in range(len(labels)):
        pred = labels[i]
        res[i][pred] = 1

    return res


def write_weights_in_file_2l(num_its, weights, file_to_save):
    file = open(file_to_save, "wb")
    np.save(file, weights)
    file.write(num_its.to_bytes(24, byteorder='big', signed=False))

    file.close()


def write_weights_in_file_3l(num_its, weights_0_1, weights_1_2, file_to_save):
    file = open(file_to_save, "wb")
    np.save(file, weights_0_1)
    np.save(file, weights_1_2)
    file.write(num_its.to_bytes(24, byteorder='big', signed=False))

    file.close()


def read_weights_from_file_2l(file_with_weights):
    file = open(file_with_weights, "rb")
    w = np.load(file)
    num_its = int.from_bytes(file.read(), byteorder='big')

    file.close()
    return w, num_its


def read_weights_from_file_3l(file_with_weights):
    file = open(file_with_weights, "rb")
    w1 = np.load(file)
    w2 = np.load(file)
    num_its = int.from_bytes(file.read(), byteorder='big')

    file.close()
    return w1, w2, num_its


def get_accuracy_of_prediction(prediction, right):
    pos_of_max_elem = np.argmax(prediction, axis=1)
    pos_of_right_max_elem = np.argmax(right, axis=1)
    equals = (pos_of_right_max_elem == pos_of_max_elem)

    true = np.count_nonzero(equals == True)
    false = np.count_nonzero(equals == False)

    percentage_of_right_answers: float = true / (true + false) * 100
    return percentage_of_right_answers


def training_nn_2_layer(file_with_weights):
    number_inputs = 784
    number_outputs = 10
    goal_predict = get_goal_predictions_in_right_form(labels)

    inputs = images
    # weights = np.random.rand(number_outputs, number_inputs)

    # epochs = 1000
    alpha = 0.005
    # i = 0

    weights, i = read_weights_from_file_2l(file_with_weights)

    while True:
        for j in range(len(images)):
            predict = np.dot(inputs[j], weights.T)
            # error = sum(sum(np.square(goal_predict - predict)))

            delta = predict - goal_predict[j]
            inp = inputs[j]

            delta = np.asmatrix(delta)
            inp = np.asmatrix(inp)

            delta_weights = alpha * np.dot(delta.T, inp)
            weights -= delta_weights

        predict = np.dot(inputs, weights.T)

        print(i)
        if i % 5 == 0:
            # print("Prediction: " + str(predict) + "\t\nWeights:\n" + str(weights) + "\nError: " + str(error) + "\n")
            print("Error: " + str(get_accuracy_of_prediction(predict, goal_predict)))

        if i % 50 == 0:
            print("----Test neural network: Err = " + str(test_nn_2_layers(weights)))

        if i % 300 == 0:
            write_weights_in_file_2l(i, weights, file_with_weights)

        i += 1


def training_nn_3_layers():
    number_inputs = 784
    number_outputs = 10
    number_hidden = 800

    np.random.seed(1)

    # weights_0_1 = 0.2 * np.random.rand(number_inputs, number_hidden) - 0.1
    # weights_1_2 = 0.2 * np.random.rand(number_hidden, number_outputs) - 0.1

    goal_predict = get_goal_predictions_in_right_form(labels)

    alpha = 0.00000001
    epochs = 1000

    weights_0_1, weights_1_2, i = read_weights_from_file_3l(file_save_weights[chosen_number])

    while True:
        ''' error, correct_cnt = (0, 0)
    
            for j in range(len(images)):
            layer_0 = images[j: j + 1]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)
    
            error += np.sum((labels[j: j + 1] - layer_2) ** 2)
            correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[j: j + 1]))
    
            layer_2_delta = labels[j: j + 1] - layer_2
            layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * relu_derivative(layer_1)
    
            weights_1_2 += alpha * np.dot(layer_1.T, layer_2_delta)
            weights_0_1 += alpha * np.dot(layer_0.T, layer_1_delta) '''

        layer_0 = images
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        error = np.sum(np.sum((layer_2 - goal_predict) ** 2))
        # correct_cnt = get_accuracy_of_prediction(layer_2, goal_predict)

        layer_2_delta = layer_2 - goal_predict
        layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * relu_derivative(layer_1)

        weights_1_2 -= alpha * np.dot(layer_1.T, layer_2_delta)
        weights_0_1 -= alpha * np.dot(layer_0.T, layer_1_delta)

        if i % 2 == 0 or i == epochs - 1:
            test_err, train_err = test_nn_3_layers(weights_0_1, weights_1_2)
            # print("I: " + str(i) + " Train-Err: " + str(error / float(len(images)))[0:5] + " Train-Acc: " +
            # str(train_err)[0:4])

            print("I: " + str(i) + " Train-Acc: " + str(train_err)[0:4] + "\tTest-Acc: " + str(test_err)[0:4] +
                  "\tError: " + str(error / float(len(images)))[0:6])

        if i % 100 == 0:
            write_weights_in_file_3l(i, weights_0_1, weights_1_2, file_save_weights[chosen_number])

        i += 1


def check_weights_2l():
    for i in file_save_weights[0:2]:
        weights_right, iters = read_weights_from_file_2l(i)
        test_err, train_err = test_nn_2_layers(weights_right)

        print(i + ":\nTraining data: " + str(train_err) + "% is correct\nTesting data: " + str(test_err) +
              "% is correct\n\n")


def check_weights_3l():
    for i in file_save_weights[2:]:
        weights_0_1, weights_1_2, iters = read_weights_from_file_3l(i)
        test_err, train_err = test_nn_3_layers(weights_0_1, weights_1_2)

        print(i + ":\nTraining data: " + str(train_err) + "% is correct\nTesting data: " + str(test_err) +
              "% is correct\n\n")


'''window = Tk()
p = Draw(window)
window.mainloop()'''

# check_weights_3l()

training_nn_3_layers()
