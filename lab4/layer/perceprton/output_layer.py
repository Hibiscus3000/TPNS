from layer.perceprton.regular_layer import *
from activation_function.activation_function import *
import numpy as np

class OutputLayer(RegularLayer):

    # z - is the output of that layer, y - expected
    def back_prop(self, z, activation_function, y):
        a = activation_function.apply(z)
        db = activation_function.derivative(z) * (a - y)
        dW = self.count_dW(db, a)
        return db, db, dW
    

