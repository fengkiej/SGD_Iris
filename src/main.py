from src.model import Model
from numpy import random
import matplotlib.pyplot as plt


def main():
    f = open("training.data")
    training_data = [[float(x) for x in line.split(" ")] for line in f]
    f.close()

    f = open("validation.data")
    validation_data = [[float(x) for x in line.split(" ")] for line in f]
    f.close()

    random.shuffle(training_data)

    model = Model(4, 0.1, prediction_label=['iris-setosa', 'iris-versicolor'])

    print("training...")
    print("inital values:\n", model.input_neuron_values)
    model.train(training_data, 60)
    print("training completed")
    print("final values:\n", model.input_neuron_values)

    print("validation...")
    model.validate(validation_data)
    print("validation completed")

    plt.plot(model.training_error_mean)
    plt.plot(model.training_error_stdev)
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.legend(['mean error', 'stdev'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
