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
        targets = self.config['targets']
        self.l = self.config['l']
        # weights of previous nn value
        self.Whh = np.random.rand(self.l, self.l)
        # weights of current input value
        self.Whx = np.random.rand(self.l, self.input_length)
        # weights of output value
        self.Who = np.random.rand(targets, self.l)
        # bias of previos and input value
        self.bh = np.random.rand(self.l, 1)
        self.bo = np.random.rand(targets, 1)
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()

    def train_all(self, learning_rate, X, Y):
        epoch = len(X) // self.window_size
        output = []
        X = np.transpose(X)
        for e in range(0, epoch):
            logging.getLogger(__name__).debug(f'TRAIN EPOCH {e} / {epoch}')
            start = e * self.window_size
            end = (e + 1) * self.window_size
            output.append(self.train(X[:, start:end], Y[start], learning_rate))

    def test_all(self, X):
        epoch = len(X) // self.window_size
        output = []
        X = np.transpose(X)
        for e in range(0, epoch):
            logging.getLogger(__name__).debug(f'TEST EPOCH {e} / {epoch}')
            start = e * self.window_size
            end = (e + 1) * self.window_size
            output.append(self.test(X[:, start:end]))
        return output

    def train(self, x, y, learning_rate):
        self.forward_prop(x)
        dbo, dWho, dbh, dWhh, dWhx = self.back_prop(y)
        self.change_weights_biases(learning_rate, dbo, dWho, dbh, dWhh, dWhx)
        return self.output

    def test(self, x):
        return self.forward_prop(x)

    def forward_prop(self, X):
        # previous nn value
        h = np.zeros((self.l, 1))
        self.X = X
        self.H = [h]
        # output
        self.Z = []
        for i in range(0, X.shape[1]):
            z = np.matmul(self.Whh, h) + np.matmul(self.Whx,
                                                   X[:, i].reshape((self.input_length, 1))) + self.bh
            self.Z.append(z)
            h = self.tanh.apply(z)
            self.H.append(h)

        self.Z = self.Z
        self.H = self.H

        self.v = np.matmul(self.Who, h) + self.bo
        self.output = self.sigmoid.apply(self.v)
        return self.output

    def back_prop(self, y):
        last_cell = len(self.Z) - 1
        dbo = (self.output - y) * self.sigmoid.derivative(self.v)
        dWho = np.matmul(dbo, np.transpose(self.H[last_cell]))
        dboWhofz_last = np.matmul(np.transpose(
            np.matmul(self.Who, self.Z[last_cell])), dbo)
        dz = self.dzdz()
        self.H = np.array(self.H).reshape(self.l, self.window_size + 1)[:,1:]
        dbh = dboWhofz_last * self.dzdbh(dz)
        dbh = dbh.reshape(self.l,1)
        dWhh = dboWhofz_last * self.dzdWhh(dz)
        dWhx = dboWhofz_last * self.dzdWhx(dz)
        return dbo, dWho, dbh, dWhh, dWhx

    def change_weights_biases(self, learning_rate, dbo, dWho, dbh, dWhh, dWhx):
        self.bo -= learning_rate * dbo
        self.Who -= learning_rate * dWho
        self.bh -= learning_rate * dbh
        self.dWhh -= learning_rate * dWhh
        self.dWhx -= learning_rate * dWhx

    def dzdz(self):
        return self.dzdz_req(np.ones((self.l, 1)))

    # list of dz[n - 1]/dz[j] where n - is the number of cells, j = 0,...,n-1
    def dzdz_req(self, grad):
        if grad.shape[1] == len(self.Z):
            return grad
        grad = np.concatenate(
            (grad, np.matmul(self.Whh, self.Z[grad.shape[1] - 1])), axis=1)
        return self.dzdz_req(grad)

    # dz[n - 1]/dbh where n - is the number of cells
    # dz - dzdz() result
    def dzdbh(self, dz):
        return np.sum(dz, axis=1)

    # dz[n - 1]/dWhh where n - is the number of cells
    # dz - dzdz() result
    def dzdWhh(self, dz):
        dzsdWhh = np.matmul(np.flip(self.H, axis=1), np.transpose(dz))
        return np.sum(dzsdWhh, axis=1)

    def dzdWhx(self, dz):
        dzsdWhx = np.matmul(np.flip(self.X, axis=1), np.transpose(dz))
        return np.sum(dzsdWhx, axis=1)
