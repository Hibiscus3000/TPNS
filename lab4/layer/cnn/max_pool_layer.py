from layer.cnn.pooling_layer import *
import numpy as np

class MaxPooling(PollingLayer):

    def __init__(self, prev_chnl, prev_height, prev_width, height, width, s1, s2):
        super.__init__(self, prev_chnl, prev_height, prev_width, height, width, s1, s2)
        self.max_inds = np.empty(self.output.shape())


    def apply_pool(self, chnl, x, y, X):
        assert(self.height, self.width == X.shape)
        self.max_inds[chnl][x][y] = np.unravel_index(np.argmax(X), X.shape)
        return np.max(X)

    def back_prop(self, next_d):
        assert(next_d.shape == self.output.shape)
        d = np.zeros((self.prev_chnl, self.prev_height, self.prev_width))
        for c in range(0, next_d.shape[0]):
            for h in range(0, next_d.shape[1]):
                for w in range(0, next_d.shape[2]):
                    d[c][h * self.s2 + self.max_inds[c][h][w][0]][w * self.s1 + self.max_inds[c][h][w][1]]\
                        = next_d[c][h][w]
        return d, None, None