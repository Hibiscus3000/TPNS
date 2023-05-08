from layer.cnn.cnn_layer import *
import numpy as np

class PollingLayer(CNNLayer):

    #  prev_chnl, prev_height, prev_width - previous layer number of chanells, height and width
    def __init__(self, prev_chnl, prev_height, prev_width, height, width, s1, s2):
        super().__init__(height, width)
        self.prev_chnl = prev_chnl
        self.prev_height = prev_height
        self.prev_width = prev_width
        self.s1 = s1
        self.s2 = s2
        self.output = np.empty(np.concatenate((np.array([prev_chnl]),
                                     np.ceil(np.array([prev_height, prev_width]) / np.array([s1,s2])).astype(int))))

    def forward_prop(self, x):
        assert self.prev_chnl == x.shape[0]
        assert self.prev_height == x.shape[1]
        assert self.prev_width == x.shape[2]
        for k in range(0, self.prev_chnl):
            for j in range(0, self.prev_height, self.s2):
                for i in range(0, self.prev_width, self.s1):
                    self.output[k][j][i] = self.apply_pool(k, j // self.s2, i // self.s1,
                                                           x[k][j:j + self.height][i:i + self.width])
        return self.output

    def change_weights_biases(self, learning_rate, db, dW):
        pass

    @abc.abstractclassmethod
    # chnl - chanel number
    # x, y - coordinate in pooling output
    # used for max pooling saving argmax 
    def apply_pool(self, chnl, x, y, X):
        pass
