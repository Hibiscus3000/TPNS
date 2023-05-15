import pandas as pd
import numpy as np
import random

number_of_classes = 10


class Reader:

    def __init__(self, train, test):
        self.train = self.read_data('data/mnist_train.csv',train)
        self.test = self.read_data('data/mnist_test.csv',test)

    def get_train(self):
        return self.train

    def get_test(self):            
        return self.test

    def read_data(self, file_name, number_of_samples):
        all_data = pd.read_csv(file_name)
        sorted_data = all_data.sort_values(all_data.columns[0]).to_numpy()
        classes_start = []
        for i in range(0, len(sorted_data)):
            if sorted_data[i][0] == len(classes_start):
                classes_start.append(i)
        classes_start.append(len(sorted_data) - 1)

        samples = np.array([])
        number_of_samples_each_class = number_of_samples // number_of_classes
        for i in range(0, number_of_classes):
            samples = np.concatenate((samples,
                                     random.sample(range(classes_start[i],
                                                         classes_start[i + 1]),
                                                   number_of_samples_each_class)))
            
        return self.proc_data(np.array([sorted_data[int(i)] for i in samples]))

    def get_random_train(self):
        sample_id = random.randint(0, len(self.train[0]) - 1)
        return (np.array([self.train[0][sample_id]]),np.array([self.train[1][sample_id]]))

    def get_random_test(self):
        sample_id = random.randint(0, len(self.test[0]) - 1)
        return (np.array([self.test[0][sample_id]]),np.array([self.test[1][sample_id]]))

    def proc_data(self, data):
        X0 = data[:, 1:] / 255
        Y0 = np.transpose(data[:, :1]).astype(int).flatten()
        X = []
        Y = []
        # one coding
        for x0 in X0:
            X.append(x0.reshape(1, 28, 28))
        for y0 in Y0:
            Y.append(self.to_one_coding(y0))
        return np.array(X), np.array(Y).astype(int)

    def to_one_coding(self, y0):
        y = np.zeros(10)
        y[y0] = 1
        return y

    def decode(self, y):
        return np.argmax(y)

    def decode_all(self, Y):
        Y0 = []
        for y in Y:
            Y0.append(self.decode(y))
        return np.array(Y0)