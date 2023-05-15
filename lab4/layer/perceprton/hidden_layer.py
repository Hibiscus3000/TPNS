from layer.perceprton.perceptron_layer import *
import numpy as np

class HiddenLayer(PerceptronLayer):

    # z - output of that layer, d - delta from previous layer
    def back_prop(self, x, z, activation_function, d):
        db = np.multiply(activation_function.derivative(z),d)
        dW = self.count_dW(db, x.flatten())
        return np.matmul(db, self.W), db, dW
