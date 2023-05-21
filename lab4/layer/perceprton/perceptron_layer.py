import numpy as np

from layer.layer import *

class PerceptronLayer(Layer):

    def __init__(self, number_of_neurons, number_of_neurons_prev_layer):
        self.W = (np.random.rand(number_of_neurons, number_of_neurons_prev_layer) - 0.5) / 5
        self.b = (np.random.rand(number_of_neurons) - 0.5) / 5

    def forward_prop(self, x):
        x = x.flatten()
        return np.matmul(self.W, x) + self.b
    
    def change_weights_biases(self, learning_rate, db, dW):
        self.b -= learning_rate * db
        self.W -= learning_rate * dW
    
    def count_dW(self, db, a):
        return np.matmul(np.transpose(np.atleast_2d(db)), np.atleast_2d(a))