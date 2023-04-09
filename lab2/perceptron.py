import numpy as np
import math


class Perceptron:
    def __init__(self, layers, neurons, learning_rate):
        self.b = []
        self.W = []
        self.layers = layers
        self.neurons = neurons
        self.learning_rate = learning_rate
        for i in range(1, layers):
            self.W[i] = np.random.randn(neurons[i], neurons[i - 1])
            self.b[i] = np.random.randn(neurons[i])

    # z = Wa + b
    def sigmoid(self, z):
        return np.ones(len(z)) / (np.ones(len(z)) + np.exp(np.negative(z)))

    def sigmoid_derivative(self, z):
        return np.multiply(self.sigmoid(z), 1 - self.sigmoid(z))

    # x - attributes
    def forward_prop(self, x, y):
        a = [x]
        z = []
        for l in range(1, self.layers):
            z.append(np.add(np.matmul(self.W[l], a[l - 1]), self.b[
                l]))
            a.append(self.sigmoid(z[l - 1]))

        C = 0  # C - cost function
        for i in range(0, len(y)):
            C = np.sum((a[self.layers] - y) ** 2)

        return C, z, a

    # z = Wa(L - 1) + b,a(L) = f(z), f - activation function, y - targets
    def back_prop(self, z, a, y):
        dW = []
        db = []

        # deltas for the output layer
        last_layer = self.layers - 1
        db.append(self.sigmoid_derivative(z[last_layer]) * (
                a[last_layer] - y))

        # deltas for hidden layers
        for i in range(last_layer - 1, 1):
            db.insert(i, self.sigmoid_derivative(z[last_layer]) * np.mult(
                db[0], self.W[i + 1]))

        for i in range(1, self.layers):
            dW.append(np.multiply(np.transpose(db), a[i - 1]))

        return dW, db

    def calc_it(self, x, y):
        C, z, a = self.forward_prop(x, y)

        dW, db = self.back_prop(z, a, y)
        for i in range(1, self.layers):
            self.W[i] -= self.learning_rate * dW[i]
            self.b[i] -= self.learning_rate * db[i]
        print(C)

    def learn(self, X, Y):
        for i in range(0, len(X)):
            self.calc_it(self, X[i], Y[i])

    def predict(self, X, Y):
        for i in range(0, len(X)):
            C, _, _ = self.forward_prop(X[i],Y[i])
            print(C)
