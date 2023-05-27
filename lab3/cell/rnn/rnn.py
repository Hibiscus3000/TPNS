import numpy as np
import json

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

    def train_all(self, X, Y):
        epoch = len(X) // self.window_size
        output = []
        for e in range(0,epoch):
            start = e * self.window_size
            end = (e + 1) * self.window_size
            output.append(self.train(X[start: end], Y[start]))

    def test_all(self, X):
        epoch = len(X) // self.window_size
        output = []
        for e in range(0, epoch):
            start = e * self.window_size
            end = (e + 1) * self.window_size
            output.append(self.test(X[start: end]))
        return output


    def train(self, x, y):
        self.forward_prop(x)
        dbo, dWho, dbh, dWhh, dWhx = self.back_prop(y)
        self.change_weights_biases(dbo, dWho, dbh, dWhh, dWhx)
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
        for x in X:
            z = np.matmul(self.Whh, h) + np.matmul(self.Whx, x.reshape(self.input_length, 1)) + self.bh
            self.Z.append(z)
            h = self.tanh.apply(z)
            self.H.append(h)

        self.v = np.matmul(self.Who, h) + self.bo
        self.output = self.sigmoid.apply(self.v)
        return self.output

    def back_prop(self, y):
        last_cell = len(self.Z) - 1
        dbo = (self.output - y) * self.sigmoid.derivative(self.v)
        dWho = np.matmul(dbo, np.transpose(self.H[last_cell]))
        dboWhofz_last = np.matmul(np.transpose(np.matmul(self.Who, self.Z[last_cell])),dbo)
        dz = self.dzdz()
        dbh = dboWhofz_last * self.dzdbh(dz)
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
        grad = np.concatenate((grad, np.matmul(self.Whh, self.Z[len(grad) - 1])))
        return self.dzdz_req(grad)

    # dz[n - 1]/dbh where n - is the number of cells
    # dz - dzdz() result
    def dzdbh(self, dz):
        return np.sum(dz, axis=0)
    
    # dz[n - 1]/dWhh where n - is the number of cells
    # dz - dzdz() result
    def dzdWhh(self, dz):
        dzsdWhh = [self.H[len(dz) - k] * dz[k] for k in range(0, len(dz))]
        return np.sum(dzsdWhh, axis=0)
    
    def dzdWhx(self, dz):
        dzsdWhx = [self.X[len(dz) - k + 1] * dz[k] for k in range(0, len(dz))]
        return np.sum(dzsdWhx, axis=0)
