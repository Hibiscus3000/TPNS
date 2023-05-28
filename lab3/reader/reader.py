import pandas as pd
import numpy as np
import json


class Reader:

    def __init__(self, window_size):
        self.mins = []
        self.deltas = []

        data = pd.read_csv('data/data.csv').to_numpy()
        with open('reader/reader.json', 'r') as config_file:
            config = json.load(config_file)
        targets = config['targets']

        attributes = []
        # test.shape[1] - number of columns
        for i in range(0, data.shape[1]):
            if i not in targets:
                attributes.append(i)

        number_of_samples = len(data)
        test_start = int(config['test_perc'] * number_of_samples)

        attributes = data[:, attributes]
        targets = self.encode(data[:, targets])

        self.train_attributes = attributes[0:test_start, :]
        train_targets = targets[0:test_start, :]
        self.test_attributes = attributes[test_start:number_of_samples, :]
        test_targets = targets[test_start:number_of_samples, :]

        # adjusting to window size
        number_of_test_samples = test_targets.shape[0]
        test_epoch = number_of_test_samples // window_size
        test_targets_ind = []

        for e in range(0, test_epoch):
            test_targets_ind.append(e * window_size)

        number_of_train_samples = train_targets.shape[0]
        train_epoch = number_of_train_samples // window_size
        train_targets_ind = []

        for e in range(0, train_epoch):
            train_targets_ind.append(e * window_size)

        self.test_targets = test_targets[test_targets_ind, :]
        self.train_targets = train_targets[train_targets_ind, :]

    def read_train(self):
        return self.train_attributes, self.train_targets

    def read_test(self):
        return self.test_attributes, self.test_targets

    def encode(self, targets):
        for i in range(0, targets.shape[1]):
            target = targets[:, i]
            minimum = np.min(target)
            delta = np.max(target) - minimum
            self.mins.append(minimum)
            self.deltas.append(delta)

        return (targets - self.mins) / self.deltas

    def decode(self, targets):
        return targets * self.deltas + self.mins
