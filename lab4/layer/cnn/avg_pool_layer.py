from layer.cnn.pooling_layer import *
import numpy as np

class AvgPoolLayer(PollingLayer):

    def apply_pool(self, chnl, x, y, X):
        assert(self.height, self.width == X.shape)
        return np.average(X)
        
    def back_prop(self, next_d):
        assert(next_d.shape == self.output.shape)
        d = np.empty((self.prev_chnl, self.prev_height, self.prev_width))
        for c in range(0, next_d.shape[0]):
            d[c] = np.sum(next_d[c]) / (self.height * self.width)
        return d, None, None