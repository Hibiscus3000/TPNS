
import random
import numpy as np
import scipy.signal

from layer.cnn.cnn_layer import *


class ConvolutionLayer(CNNLayer):

    # image_depth - number of input channels
    # filters - number_of_filters
    # p - padding
    def __init__(self, image_depth, size, filters, p):
        super().__init__(size, image_depth)
        self.filters = filters
        self.W = tuple([tuple([np.random.rand(size, size) - 0.5
                       for i in range(0, image_depth)]) for j in range(0, filters)])
        self.b = tuple([random.random() - 0.5 for f in range(0, filters)])
        self.p = p

    def forward_prop(self, X):
        X = tuple(np.pad(X, ((0, 0), (self.p, self.p), (self.p, self.p)),
                         constant_values=(0)))
        output_size = self.calc_output_size(X[0], self.size)
        output = list(np.zeros((self.filters, output_size, output_size)))
        for f in range(0, self.filters):
            for j in range(0, self.image_depth):
                output[f] += self.convolve(X[j], self.W[f][j])
            output[f] += self.b[f]
        return np.array(output)

    def back_prop(self, X, next_d):
        X = tuple(np.pad(X, ((0, 0), (self.p, self.p), (self.p, self.p)),
                         constant_values=(0)))
        db = [np.sum(next_d[f]) for f in range(0, self.filters)]
        dW = [[self.convolve(X[j], next_d[f]) for j in range(0, self.image_depth)]
              for f in range(0, self.filters)]
        padding_size = next_d[0].shape[0] - 1
        d = [np.sum([self.convolve(np.pad(self.W[f][j], (padding_size, padding_size),
                                     constant_values=(0))[::-1, ::-1],
                              next_d[f]) for f in range(0, self.filters)], axis=(0)) for j in range(0, self.image_depth)]
        return np.array(d), np.array(db), np.array(dW)

    def change_weights_biases(self, learning_rate, db, dW):
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

    # A - hxw
    # K - khxkw
    def convolve(self, A, K):
        return scipy.signal.convolve(A, np.flip(K), mode='valid')

    # k_size - kernel size
    def calc_output_size(self, A, k_size):
        h, w = A.shape[-2:]
        # assert h == w
        k = k_size // 2
        add = k_size % 2
        output_size = (h - 2 * k + 1 - add)
        return output_size
