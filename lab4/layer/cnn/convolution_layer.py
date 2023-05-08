from layer.cnn.cnn_layer import *
import numpy as np


class ConvolutionLayer(CNNLayer):

    # image_depth - number of input channels
    # filters - number_of_filters
    # p - padding
    def __init__(self, image_depth, height, width, filters, p):
        super().__init__(height, width)
        self.filters = filters
        self.W = np.random.rand(filters, image_depth, height, width) - 0.5
        self.b = np.random.rand(filters) - 0.5
        self.p = p

    def forward_prop(self, X):
        output = np.zeros((self.fitters, self.height, self.width))
        for j in range(0, self.image_depth):
            X[j] = np.pad(X[j], (self.p, self.p), constant_values=(0))
        for f in range(0, self.filters):
            for j in range(0, self.image_depth):
                output[f] += self.convolve(X[j], self.W[f][j])
            output[f] += self.bias[f]
        return output

    def back_prop(self, X, d):
        for j in range(0, self.image_depth):
            X[j] = np.pad(X[j], (self.p, self.p), constant_values=(0))
        db = np.empty(self.filters)
        dW = np.empty(self.W.shape)
        d = np.zeros(X.shape)
        for f in range(0, self.filters):
            db[f] = np.sum(d[f])
            for j in range(0, self.image_depth):
                dW[f][j] = self.convolve(X[j], db[f])
                d[j] += self.convolve(np.rot90(np.rot90(np.pad(dW[f][j], (d.shape[0] - 1,d.shape[0] - 1), constant_values=(0)))), db[f])
        return d, db, dW


    def change_weights_biases(self, learning_rate, db, dW):
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

    # A - hxw
    # K - khxkw
    def convolve(self, A, K, s1, s2):
        h, w = A.shape
        kh, kw = K.shape
        k1 = kh // 2 - 1
        k2 = kw // 2 - 1
        output_h = (h - 2 * k1)
        output_w = (w - 2 * k2)
        output = np.empty((output_h, output_w))
        for y in range(k1, h - k1):
            for x in range(k2, w - k2):
                output[y - k1][x - k2] = np.sum(A[y - k1: y + k1 + 1][x - k2: x + k2 + 1] * K)
        return output
