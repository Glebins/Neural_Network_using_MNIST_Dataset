import numpy as np
import random


class NeuralNetwork:

    def __int__(self, train_images, train_labels, test_images, test_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

    def test_nn_2_layers(self, weights):
        test_right_answers = self.get_goal_predictions_in_right_form(self.test_labels)

        test_predictions = np.dot(self.test_images, weights.T)
        test_error = self.get_accuracy_of_prediction(test_predictions, test_right_answers)

        goal_predict = self.get_goal_predictions_in_right_form(self.train_labels)

        prediction_on_training_data = np.dot(self.train_images, weights.T)
        error_training_data = self.get_accuracy_of_prediction(prediction_on_training_data, goal_predict)

        return test_error, error_training_data

    def test_nn_3_layers(self, weights_0_1, weights_1_2):

        test_right_answers = self.get_goal_predictions_in_right_form(self.test_labels)

        layer_1_test = relu(np.dot(self.test_images, weights_0_1))
        test_predictions = np.dot(layer_1_test, weights_1_2)
        test_error = self.get_accuracy_of_prediction(test_predictions, test_right_answers)

        train_right_answers = self.get_goal_predictions_in_right_form(self.train_labels)

        layer_1_train = relu(np.dot(self.train_images, weights_0_1))
        train_prediction = np.dot(layer_1_train, weights_1_2)
        train_error = self.get_accuracy_of_prediction(train_prediction, train_right_answers)

        return test_error, train_error

    def get_predict(self, inputs, weights):
        predict = np.dot(inputs, weights.T)
        return predict

    def print_random_image_from_dataset(self):
        index = random.randrange(0, len(self.train_images))

        for i in range(28):
            for j in range(28):
                print(self.train_images[index][28 * i + j], end="\t")

            print()

        print("\n\n", self.train_labels[index])

    def get_goal_predictions_in_right_form(self, labels):
        res = np.zeros((len(labels), 10))

        for i in range(len(labels)):
            pred = labels[i]
            res[i][pred] = 1

        return res

    def write_weights_in_file_2l(self, num_its, weights, file_to_save):
        file = open(file_to_save, "wb")
        np.save(file, weights)
        file.write(num_its.to_bytes(24, byteorder='big', signed=False))

        file.close()

    def write_weights_in_file_3l(self, num_its, weights_0_1, weights_1_2, file_to_save):
        file = open(file_to_save, "wb")
        np.save(file, weights_0_1)
        np.save(file, weights_1_2)
        file.write(num_its.to_bytes(24, byteorder='big', signed=False))

        file.close()

    def read_weights_from_file_2l(self, file_with_weights):
        file = open(file_with_weights, "rb")
        w = np.load(file)
        num_its = int.from_bytes(file.read(), byteorder='big')

        file.close()
        return w, num_its

    def read_weights_from_file_3l(self, file_with_weights):
        file = open(file_with_weights, "rb")
        w1 = np.load(file)
        w2 = np.load(file)
        num_its = int.from_bytes(file.read(), byteorder='big')

        file.close()
        return w1, w2, num_its

    def get_accuracy_of_prediction(self, prediction, right):
        pos_of_max_elem = np.argmax(prediction, axis=1)
        pos_of_right_max_elem = np.argmax(right, axis=1)
        equals = (pos_of_right_max_elem == pos_of_max_elem)

        true = np.count_nonzero(equals == True)
        false = np.count_nonzero(equals == False)

        percentage_of_right_answers: float = true / (true + false) * 100
        return percentage_of_right_answers

    def training_nn_2_layer(self, file_with_weights):
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

    def training_nn_3_layers(self):
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

    def check_weights_2l(self):
        for i in file_save_weights[0:2]:
            weights_right, iters = read_weights_from_file_2l(i)
            test_err, train_err = test_nn_2_layers(weights_right)

            print(i + ":\nTraining data: " + str(train_err) + "% is correct\nTesting data: " + str(test_err) +
                  "% is correct\n\n")

    def check_weights_3l(self):
        for i in file_save_weights[2:]:
            weights_0_1, weights_1_2, iters = read_weights_from_file_3l(i)
            test_err, train_err = test_nn_3_layers(weights_0_1, weights_1_2)

            print(i + ":\nTraining data: " + str(train_err) + "% is correct\nTesting data: " + str(test_err) +
                  "% is correct\n\n")


def relu(x):
    return (x > 0) * x


def relu_derivative(x):
    return x > 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)
