import numpy as np
import skimage
import math

from layer.cnn.cnn_layer import *


class PollingLayer(CNNLayer):

    #  image_depth, prev_size - previous layer number of chanells, height and width
    def __init__(self, image_depth, prev_size, size, stride):
        super().__init__(size, image_depth)
        self.prev_size = prev_size
        self.stride = stride
        self.output_size = self.get_output_size()

    def forward_prop(self, x):
        # assert (self.image_depth, self.prev_size, self.prev_size) == x.shape
        if (self.stride == self.size) & (0 == self.prev_size % self.size):
            return skimage.measure.block_reduce(x, (1,self.size,self.size), np.average)
        else:
            output = np.empty(self.output_size)
            for k in range(0, self.image_depth):
                for j in range(0, self.prev_size, self.stride):
                    for i in range(0, self.prev_size, self.stride):
                        output[k][j // self.stride][i // self.stride] =\
                            self.apply_pool(k, j // self.stride, i // self.stride,
                                            x[k, j:j + self.size, i:i + self.size])
            return output

    def change_weights_biases(self, learning_rate, db, dW):
        pass

    def get_output_size(self):
        output_size = int(math.ceil(self.prev_size / self.stride))
        return np.concatenate((np.array([self.image_depth]), np.array([output_size, output_size])))

    @abc.abstractclassmethod
    # chnl - chanel number
    # x, y - coordinate in pooling output
    # used for max pooling saving argmax
    def apply_pool(self, chnl, x, y, X):
        pass
