from layer.cnn.pooling_layer import *

import numpy as np

class AvgPoolLayer(PollingLayer):

    def apply_pool(self, chnl, x, y, X):
        # assert (self.size, self.size) == X.shape
        return np.average(X)
        
    def back_prop(self, next_d):
        d = np.zeros((self.image_depth, self.prev_size, self.prev_size))
        next_d = next_d.reshape(self.output_size)
        N = self.size * self.size
        for c in range(0, next_d.shape[0]):
            for y in range(0, next_d.shape[1]):
                for x in range(0, next_d.shape[2]):
                    d_y = y * self.stride
                    d_x = x * self.stride
                    d[c, d_y : d_y + self.size, d_x : d_x + self.size] += next_d[c][y][x] / N
        return d, None, None