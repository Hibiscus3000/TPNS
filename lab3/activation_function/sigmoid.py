from activation_function.activation_function import *
import numpy as np

class Sigmoid(ActivationFunction):

    def apply(self, z):
        return 1 / (1 + np.exp(-z))
    
    def derivative(self, z):
        value = self.apply(z)
        return np.multiply(value, 1 - value)