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

        # adjusting to window size
        test_start = (test_start // window_size) * window_size
        test_end = (number_of_samples // window_size) * window_size

        self.train_attributes = attributes[0:test_start, :]
        self.train_targets = targets[0:test_start, :]
        self.test_attributes = attributes[test_start:test_end, :]
        self.test_targets = targets[test_start:test_end, :]

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
