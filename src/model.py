from numpy import random
from math import exp
from statistics import stdev, mean


class Model:
    def __init__(self, neurons, learning_rate, prediction_label=None, threshold=0.5):
        if prediction_label is None:
            prediction_label = ['0', '1']
        self.alpha = learning_rate
        self.input_neurons = neurons
        self.input_neuron_values = random.rand(neurons + 1, 1)
        self.prediction_label = prediction_label
        self.threshold = threshold
        self.training_error_mean = []
        self.training_error_stdev = []

    def train(self, training_dataset, epoch):
        for i in range(0, epoch):
            random.shuffle(training_dataset)
            error_dataset = []
            for data in training_dataset:
                prediction = self.predict(data)

                fact = data[-1]
                error = (prediction - fact) ** 2
                error_dataset.append(error)
                self.update_model(prediction, fact, data)

            self.training_error_mean.append(mean(error_dataset))
            self.training_error_stdev.append(stdev(error_dataset))

    def update_model(self, prediction, fact, data):
        self.input_neuron_values[-1] -= self.alpha * self.delta_(prediction, fact, 1)  # update bias
        for x in range(0, self.input_neurons):  # update theta
            self.input_neuron_values[x] -= self.alpha * self.delta_(prediction, fact, data[x])

    def predict(self, data):
        ev = 0
        ev += self.input_neuron_values[-1]  # add bias
        for x in range(0, self.input_neurons):  # theta*input
            ev += self.input_neuron_values[x] * data[x]
        prediction = self.sigmoid_activation(ev)

        return prediction

    def validate(self, validation_dataset):
        random.shuffle(validation_dataset)
        right_guess = 0
        for data in validation_dataset:
            ans = 0
            prediction = self.predict(data)
            if prediction >= self.threshold:
                ans = 1

            if ans == data[-1]:
                right_guess += 1
        print("accuracy:", right_guess / len(validation_dataset) * 100, '%')

    @staticmethod
    def sigmoid_activation(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def delta_(prediction, fact, x):
        return 2 * (prediction - fact) * (1 - prediction) * prediction * x

    @staticmethod
    def show_label(self, prediction):
        if prediction < self.threshold:
            return self.prediction_label[0]
        else:
            return self.prediction_label[1]
