from layer import *
from activation_function import *

class CNN():

    def __init__(self):
        self.layers = []
        self.activation_functions = []

    def add_layer(self, layer):
        self.layers.append(layer)
    
    def add_activation_function(self, activation_function):
        self.activation_functions.append(activation_function)