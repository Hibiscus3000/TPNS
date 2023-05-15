import time
import logging

from layer import *
from activation_function import *

class CNN():

    def __init__(self):
        self.parts = []

    def add_part(self, nn_part):
        self.parts.append(nn_part)

    def forward_prop(self, x):
        output = x
        for nn_part in self.parts:
            output = nn_part.forward_prop(output)
        return output

    def back_prop(self, learning_rate, y):
        d = y
        self.dbs = []
        self.dWs = []
        for i in range(len(self.parts) - 1, -1, -1):
            d, db, dW = self.parts[i].back_prop(d)
            self.dbs.insert(0,db)
            self.dWs.insert(0,dW)
        
        for i in range(0, len(self.parts)):
            self.parts[i].change_weights_biases(learning_rate, self.dbs[i], self.dWs[i])
        
    def train(self, learning_rate, X, Y):
        assert len(X) == len(Y)
        outputs = []
        for i in range(0, len(X)):
            t1 = time.time()
            outputs.append(self.forward_prop(X[i]))
            t2 = time.time()
            self.back_prop(learning_rate, Y[i])
            t3 = time.time()
            logging.getLogger(__name__).debug(f'{i + 1}/{len(X)}: forth - {t2 - t1} back - {t3 - t2} s')
        return outputs

    def rollback_prev_training(self, learning_rate):
        for i in range(0, len(self.parts)):
            self.parts[i].change_weights_biases(-learning_rate, self.dbs[i], self.dWs[i])

    def test(self, X):
        outputs = []
        for i in range(0, len(X)):
            outputs.append(self.forward_prop(X[i]))
        return outputs