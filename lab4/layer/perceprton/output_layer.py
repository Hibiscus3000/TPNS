from layer.perceprton.perceptron_layer import *
from activation_function.activation_function import *
import numpy as np


class OutputLayer(PerceptronLayer):

    # x - is the input of that layer
    # z - is the output of that layer
    # y - expected result
    def back_prop(self, x, z, activation_function, y):
        a = activation_function.apply(z)
        db = activation_function.derivative(z) * (a - y)
        dW = self.count_dW(db, x.flatten())
        return np.matmul(db, self.W), db, dW
