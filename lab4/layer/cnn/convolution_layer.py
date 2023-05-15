from layer.cnn.cnn_layer import *
import numpy as np


class ConvolutionLayer(CNNLayer):

    # image_depth - number of input channels
    # filters - number_of_filters
    # p - padding
    def __init__(self, image_depth, size, filters, p):
        super().__init__(size, image_depth)
        self.filters = filters
        self.W = np.random.rand(filters, image_depth, size, size) - 0.5
        self.b = np.random.rand(filters) - 0.5
        self.p = p

    def forward_prop(self, X):
        X = np.pad(X, ((0, 0), (self.p, self.p), (self.p, self.p)),
                   constant_values=(0))
        output_size = self.calc_output_size(X, self.size)
        output = np.zeros((self.filters, output_size, output_size))
        for f in range(0, self.filters):
            for j in range(0, self.image_depth):
                output[f] += self.convolve(X[j], self.W[f][j])
            output[f] += self.b[f]
        return output

    def back_prop(self, X, next_d):
        X = np.pad(X, ((0, 0), (self.p, self.p), (self.p, self.p)),
                   constant_values=(0))
        db = np.empty(self.filters)
        dW = np.empty(self.W.shape)
        d = np.zeros(X.shape)
        for f in range(0, self.filters):
            db[f] = np.sum(next_d[f])
            for j in range(0, self.image_depth):
                dW[f][j] = self.convolve(X[j], next_d[f])
                d[j] += self.convolve(np.rot90(np.rot90(np.pad(self.W[f][j],
                                                        (next_d[f].shape[0] - 1, next_d[f].shape[0] - 1),
                                                        constant_values=(0)))), next_d[f])
        return d, db, dW

    def change_weights_biases(self, learning_rate, db, dW):
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

    # A - hxw
    # K - khxkw
    def convolve(self, A, K):
        h, w = A.shape
        k_size = K.shape[0]
        k = k_size // 2
        output_size = self.calc_output_size(A, k_size)
        output = np.empty((output_size, output_size))
        add = k_size % 2
        for y in range(k, h - k + 1 - add):
            for x in range(k, w - k + 1 - add):
                output[y - k][x - k] = np.sum(
                    A[y - k: y + k + add, x - k: x + k + add] * K)
        return output

    # k_size - kernel size
    def calc_output_size(self, A, k_size):
        h, w = A.shape[-2:]
        assert h == w
        k = k_size // 2
        add = k_size % 2
        output_size = (h - 2 * k + 1 - add)
        return output_size
