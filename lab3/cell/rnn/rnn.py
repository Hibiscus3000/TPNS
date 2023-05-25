import numpy as np
import json

from cell.cell import *
from activation_function import *


class RNN(Cell):

    def __init__(self):
        with open('rnn.json', 'r') as config_file:
            self.config = json.load(config_file)
        input_length = self.config['input_length']
        targets = self.config['targets']
        self.l = self.config['l']
        # weights of previous nn value
        self.Whh = np.random.rand(self.l, self.l)
        # weights of current input value
        self.Whx = np.random.rand(self.l, input_length)
        # weights of output value
        self.Who = np.random.rand(targets, self.l)
        # bias of previos and input value
        self.bh = np.random.rand(self.l, 1)
        self.bo = np.random.rand(targets, 1)
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()

    def forward_prop(self, X):
        # previous nn value
        h = np.zeros(self.l, 1)
        self.H = [h]
        # output
        self.Z = []
        for x in X:
            z = np.matmul(self.Whh, h) + np.matmul(self.Whx, x) + self.bh
            self.Z.append(z)
            h = self.tanh.apply(z)
            self.H.append(h)

        self.v = np.matmul(self.Who, h) + self.bo
        self.output = self.sigmoid.apply(self.v)
        return self.output

    def back_prop(self, y):
        last_cell = len(self.Z) - 1
        # n1 - n0
        dn = len(self.Z)
        dbo = (self.output - y) * self.sigmoid.derivative(self.v)
        dWho = np.matmul(dbo, np.transpose(self.H[last_cell]))
        Whodfz_last = np.matmul(self.Who, self.tanh.derivate(self.Z[last_cell]))
        dbh = np.transpose(Whodfz_last * (dn * np.ones(self.self.l, 1)
                              + np.sum([np.sum([np.prod([np.matmult(self.Whh, self.tanh.derivative(self.Z[j]))
                                                                                      for j in range(i, n - 1)])
                                                                             for i in range(0, n - 1)])
                                                                     for n in range(0, dn)]))) * dbo
        dWhx = np.transpose(Whodfz_last * np.sum([np.tanh.apply(self.Z[n]) + np.sum([np.tanh.apply(self.Z[i]) * np.prod([np.matmult(self.Whh, self.tanh.derivative(self.Z[j]))
                                                                                      for j in range(i, n - 2)])
                                                                             for i in range(0, n - 1)]) + np.prod([np.matmult(self.Whh, self.tanh.derivative(self.Z[j]))
                                                                                      for j in range(0, n - 1)])
                                                                     for n in range(0, dn)])) * dbo
