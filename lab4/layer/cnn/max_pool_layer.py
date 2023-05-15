from layer.cnn.pooling_layer import *
import numpy as np


class MaxPooling(PollingLayer):

    def __init__(self, prev_chnl, prev_size, size, stride):
        super.__init__(self, prev_chnl, prev_size, size, stride)
        self.max_inds = np.empty(self.get_output_size())

    def apply_pool(self, chnl, x, y, X):
        assert self.size, self.size == X.shape
        self.max_inds[chnl][x][y] = np.unravel_index(np.argmax(X), X.shape)
        return np.max(X)

    def back_prop(self, next_d):
        d = np.zeros((self.image_depth, self.prev_size, self.prev_size))
        for c in range(0, next_d.shape[0]):
            for h in range(0, next_d.shape[1]):
                for w in range(0, next_d.shape[2]):
                    d[c][h * self.stride + self.max_inds[c][h][w][0]][w * self.stride + self.max_inds[c][h][w][1]]\
                        = max(next_d[c][h][w], d[c][h * self.stride + self.max_inds[c][h][w][0]]
                              [w * self.stride + self.max_inds[c][h][w][1]])
        return d, None, None
