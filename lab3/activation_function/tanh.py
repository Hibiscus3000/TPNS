from activation_function.activation_function import *
import numpy as np

class Tanh(ActivationFunction):

    def apply(self, z):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)
    
    def derivative(self, z):
        return 1 - self.apply(z) ** 2