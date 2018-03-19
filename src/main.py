from src.model import Model
from numpy import random


def main():
    f = open("training.data")
    training_data = [[float(x) for x in line.split(" ")] for line in f]
    f.close()

    f = open("validation.data")
    validation_data = [[float(x) for x in line.split(" ")] for line in f]
    f.close()

    random.shuffle(training_data)

    model = Model(4, 0.8, prediction_label=['iris-setosa', 'iris-versicolor'])
    model.train(training_data, 60)

    print(model.show_label(model.predict([6.3, 2.3, 4.4, 1.3, 1])))
    model.validate(validation_data)


if __name__ == '__main__':
    main()
