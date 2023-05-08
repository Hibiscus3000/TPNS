from layer.perceprton.regular_layer import *
import numpy as np

class HiddenLayer(RegularLayer):

    # z - output of that layer, d - delta from previous layer
    def back_prop(self, z, d, activation_function):
        db = np.multiply(activation_function.derivative(z),np.matmul(d,self.W))
        dW = self.count_dW(db, activation_function.derivative(z))
        return db, db, dW
