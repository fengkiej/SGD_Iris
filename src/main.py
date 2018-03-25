from src.model import Model
import matplotlib.pyplot as plt


def shift(seq, shift=1):
    return seq[-shift:] + seq[:-shift]


def main():
    f = open("iris.dataset")
    dataset = [[float(x) for x in line.split(" ")] for line in f]
    dataset = dataset[0:100]
    f.close()

    # k-fold cross validation
    k = 5
    fold = []
    fold_len = int(len(dataset) / k)
    print(dataset)
    for i in range(0, k):
        r = {'validation_data': dataset[0:fold_len], 'training_data': dataset[fold_len:len(dataset)]}
        fold.append(r)
        dataset = shift(dataset, -fold_len)

    model = Model(4, 0.1, prediction_label=['iris-setosa', 'iris-versicolor'])

    for i in range(0, k):
        print("training...")
        print("inital values:\n", model.input_neuron_values)
        model.train(fold[i]['training_data'], fold[i]['validation_data'], 60)
        print("training completed")
        print("final values:\n", model.input_neuron_values)

        plt.plot(model.training_error_mean)
        plt.plot(model.validation_result_per_epoch)
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend(['MSE per epoch', 'Validation result per epoch'], loc='upper right')
        plt.title("fold"+str(i+1)+" processed")
        plt.show()


if __name__ == '__main__':
    main()
