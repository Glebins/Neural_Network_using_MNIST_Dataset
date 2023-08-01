import numpy as np
from mnist import MNIST
import random

import math
from tkinter import *
from tkinter.ttk import Scale

file_save_weights = ["weights.txt", "weights_1.txt", "weights_2.txt", "weights_some_new.txt",
                     "weights_speed_is_more.txt"]

dataset = MNIST('C:/Users/nedob/Programming/Data Science/Datasets/MNIST_handwritten_numbers/archive')
images, labels = dataset.load_training()
images, labels = np.array(images), np.array(labels)


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


# Defining Class and constructor of the Program
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

        ''' self.background.create_line(20, 0, 20, 300, fill="green", width=0)
        self.background.create_line(40, 0, 40, 300, fill="yellow", width=0)
        self.background.create_line(60, 0, 60, 300, fill="green", width=0) '''

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

        self.infographic.create_text(20, 30, text="0", fill="black", font='Helvetica 15 bold')
        self.infographic.create_text(20, 60, text="1", fill="black", font='Helvetica 15 bold')
        self.infographic.create_text(20, 90, text="2", fill="black", font='Helvetica 15 bold')
        self.infographic.create_text(20, 120, text="3", fill="black", font='Helvetica 15 bold')
        self.infographic.create_text(20, 150, text="4", fill="black", font='Helvetica 15 bold')
        self.infographic.create_text(20, 180, text="5", fill="black", font='Helvetica 15 bold')
        self.infographic.create_text(20, 210, text="6", fill="black", font='Helvetica 15 bold')
        self.infographic.create_text(20, 240, text="7", fill="black", font='Helvetica 15 bold')
        self.infographic.create_text(20, 270, text="8", fill="black", font='Helvetica 15 bold')
        self.infographic.create_text(20, 300, text="9", fill="black", font='Helvetica 15 bold')

        max_len = 300
        alpha = max_len / np.max(predict)

        for i in range(0, 10):
            self.infographic.create_line(50, 30 * (i + 1), alpha * predict[i] + 50, 30 * (i + 1), fill="green", width=0)

        self.result = Canvas(self.root, bg='white', bd=5, relief=GROOVE, height=40, width=40)
        self.result.place(x=500, y=400)
        self.result.create_text(30, 30, text=str(np.argmax(predict)), fill="black", font='Helvetica 15 bold')

    def guess_number(self):
        weights_right, iters = read_weights_from_file(file_save_weights[2])
        pixels = compress_array(self.pixels).T
        pixels = np.reshape(pixels, (1, 784))

        pixels = pixels.astype(int)

        for i in range(28):
            for j in range(28):
                print(pixels[0][28 * i + j], end="\t")
            print()

        predict = get_predict(pixels[0], weights_right)
        self.draw_infographic(predict)
        print(predict, np.argmax(predict), sep="\n\n", end="\n")


def test_nn(weights):
    test_images, test_labels = dataset.load_testing()
    test_images, test_labels = np.array(test_images), np.array(test_labels)
    test_right_answers = get_goal_predictions_in_right_form(test_labels)

    test_predictions = np.dot(test_images, weights.T)
    test_error = get_accuracy_of_prediction(test_predictions, test_right_answers)

    goal_predict = get_goal_predictions_in_right_form(labels)

    prediction_on_training_data = np.dot(images, weights.T)
    error_training_data = get_accuracy_of_prediction(prediction_on_training_data, goal_predict)

    return test_error, error_training_data


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


def write_weights_in_file(num_its, weights, file_to_save):
    file = open(file_to_save, "wb")
    np.save(file, weights)
    file.write(num_its.to_bytes(24, byteorder='big', signed=False))

    file.close()


def read_weights_from_file(file_with_weights):
    file = open(file_with_weights, "rb")
    w = np.load(file)
    num_its = int.from_bytes(file.read(), byteorder='big')

    file.close()
    return w, num_its


def get_accuracy_of_prediction(prediction, right):
    pos_of_max_elem = np.argmax(prediction, axis=1)
    pos_of_right_max_elem = np.argmax(right, axis=1)
    equals = (pos_of_right_max_elem == pos_of_max_elem)

    true = np.count_nonzero(equals == True)
    false = np.count_nonzero(equals == False)

    percentage_of_right_answers: float = true / (true + false) * 100
    return percentage_of_right_answers


def training_nn():
    number_inputs = 784
    number_outputs = 10
    goal_predict = get_goal_predictions_in_right_form(labels)

    inputs = images
    weights = np.random.rand(number_outputs, number_inputs)

    # epochs = 1000
    alpha = 0.00000001
    i = 0

    # weights, i = read_weights_from_file()

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
            print("----Test neural network: Err = " + str(test_nn(weights)))

        if i % 300 == 0:
            write_weights_in_file(i, weights, file_save_weights[3])

        i += 1


def check_weights():
    for i in file_save_weights:
        weights_right, iters = read_weights_from_file(i)
        test_err, train_err = test_nn(weights_right)

        print(i + ":\nTraining data: " + str(train_err) + "% is correct\nTesting data: " + str(test_err) +
              "% is correct\n\n")


window = Tk()
p = Draw(window)
window.mainloop()
