from activation_function.activation_function import *
import numpy as np

class Sigmoid(ActivationFunction):

    def apply(self, z):
        return 1 / (1 + np.exp(-z))
    
    def derivative(self, z):
        return np.multiply(self.apply(z), 1 - self.apply(z))