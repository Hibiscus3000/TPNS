import numpy as np
import json
import logging

from cell.cell import *
from activation_function import *


class RNN(Cell):

    def __init__(self, window_size):
        super().__init__(window_size)
        with open('cell/rnn/rnn.json', 'r') as config_file:
            self.config = json.load(config_file)
        self.input_length = self.config['input_length']
        self.targets = self.config['targets']
        self.l = self.config['l']
        # weights of previous nn value
        self.Whh = np.random.rand(self.l, self.l) - 0.5
        # weights of current input value
        self.Whx = np.random.rand(self.l, self.input_length) - 0.5
        # weights of output value
        self.Who = np.random.rand(self.targets, self.l) - 0.5
        # bias of previos and input value
        self.bh = np.random.rand(self.l, 1) - 0.5
        self.bo = np.random.rand(self.targets, 1) - 0.5
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()

    def train_all(self, learning_rate, X, Y):
        epoch = len(X) // self.window_size
        output = []
        X = np.transpose(X)
        Y = np.transpose(Y)
        for e in range(0, epoch):
            logging.getLogger(__name__).debug(f'TRAIN EPOCH {e + 1} / {epoch}')
            start = e * self.window_size
            end = (e + 1) * self.window_size
            output.append(self.train(X[:, start:end], Y[:, start:end], learning_rate))
        return np.transpose(np.array(output).reshape(Y.shape))

    def test_all(self, X, output_shape):
        epoch = len(X) // self.window_size
        output = []
        X = np.transpose(X)
        for e in range(0, epoch):
            logging.getLogger(__name__).debug(f'TEST EPOCH {e} / {epoch}')
            start = e * self.window_size
            end = (e + 1) * self.window_size
            output.append(self.test(X[:, start:end]))
        return np.transpose(np.array(output).reshape(output_shape))

    def train(self, x, y, learning_rate):
        self.forward_prop(x)
        dbo, dWho, dbh, dWhh, dWhx = self.back_prop(y)
        self.change_weights_biases(learning_rate, dbo, dWho, dbh, dWhh, dWhx)
        return self.O

    def test(self, x):
        return self.forward_prop(x)

    def forward_prop(self, X):
        # input
        self.X = X
        # memory
        self.Z = []
        # h = tanh(z)
        h = np.zeros((self.l, 1))
        self.H = [h]
        # output 
        self.V = []
        # o = sigmoid(v)
        self.O = []

        for i in range(0, X.shape[1]):
            z = np.matmul(self.Whh, h) + np.matmul(self.Whx,
                                                   X[:, i].reshape((self.input_length, 1))) + self.bh
            self.Z.append(z)

            h = self.tanh.apply(z)
            self.H.append(h)

            v = np.matmul(self.Who, h) + self.bo
            self.V.append(v)

            o = self.sigmoid.apply(v)
            self.O.append(o)

        return self.O

    def back_prop(self, Y):
        dWhx, dWhh, dWho = np.zeros_like(self.Whx), np.zeros_like(self.Whh), np.zeros_like(self.Who)
        dbh, dbo = np.zeros_like(self.bh), np.zeros_like(self.bo)
        dh_prev = np.zeros_like(self.H[0])
        for t in range(0, len(self.X)):
            do = (self.O[t] - Y[:, t].reshape((self.targets, 1))) * self.sigmoid.derivative(self.V[t])
            dWho += np.matmul(do, np.transpose(self.H[t]))
            dbo += do
            dh = np.dot(np.transpose(self.Who), do) + dh_prev
            # dC/dz[t] (C - cost function)
            dCdz = self.tanh.derivative(self.Z[t]) * dh
            dbh += dCdz
            dWhx += np.matmul(dCdz, np.transpose(self.X[:, t].reshape((self.input_length, 1))))
            dWhh += np.matmul(dCdz, np.transpose(self.H[t - 1]))
            dh_prev = np.matmul(np.transpose(self.Whh), dCdz)

        return dbo, dWho, dbh, dWhh, dWhx

    def change_weights_biases(self, learning_rate, dbo, dWho, dbh, dWhh, dWhx):
        self.bo -= learning_rate * dbo
        self.Who -= learning_rate * dWho
        self.bh -= learning_rate * dbh
        self.Whh -= learning_rate * dWhh
        self.Whx -= learning_rate * dWhx
