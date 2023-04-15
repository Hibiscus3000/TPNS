import numpy as np
from logging import *
from formater import *

class Perceptron:
    def __init__(self, neurons):
        self.layers = len(neurons)
        self.b = []
        self.W = []
        for i in range(1, self.layers):
            self.W.append(np.random.rand(neurons[i], neurons[i - 1]) - 0.5)
            self.b.append(np.random.rand(neurons[i]) - 0.5)
        getLogger(__name__).debug("perceptron initialized")
        self.log_weights_biases()

    # z = Wa + b
    def sigmoid(self, z):
        return np.ones(len(z)) / (np.ones(len(z)) + np.exp(np.negative(z)))

    def sigmoid_derivative(self, z):
        return np.multiply(self.sigmoid(z), 1 - self.sigmoid(z))

    # x - attributes
    def forward_prop(self, x):
        a = [x]
        z = []
        for l in range(1, self.layers):
            z.append(np.add(np.matmul(self.W[l], a[l - 1]), self.b[
                l]))
            a.append(self.sigmoid(z[l - 1]))
        return z, a

    # output - output layer results
    def cost(self, output, y):
        C = 0  # C - cost function
        for i in range(0, len(y)):
            if (y[i] is not None):
                C += (output[i] - y[i]) ** 2

    def cost_component(self, output, y):
        return [output[i] - y[i] if y[i] is not None else None for i in
                range(0, len(y))]

    # z = Wa(L - 1) + b,a(L) = f(z), f - activation function, y - targets
    def back_prop(self, z, a, y):
        dW = []
        db = []

        # deltas for the output layer
        last_layer = self.layers - 1
        y_no_nones = [y[i] if y[i] is not None else a[last_layer][i] for i in
                      range(0, len(y))]
        db.append(self.sigmoid_derivative(z[last_layer]) * (
                a[last_layer] - y_no_nones))

        # deltas for hidden layers
        for i in range(last_layer - 1, 1):
            db.insert(i, self.sigmoid_derivative(z[last_layer]) * np.mult(
                db[0], self.W[i + 1]))

        for i in range(1, self.layers):
            dW.append(np.multiply(np.transpose(db), a[i - 1]))

        return dW, db

    def learn_it(self, x, y, learning_rate):
        z, a = self.forward_prop(x)

        dW, db = self.back_prop(z, a, y)
        for i in range(1, self.layers):
            self.W[i] -= learning_rate * dW[i]
            self.b[i] -= learning_rate * db[i]
        return a[self.layers - 1]

    def learn(self, X, Y, learning_rate):
        i = 0
        for sample_id, x in X:
            output = self.learn_it(X, Y[sample_id], learning_rate)
            self.log_it_results(DEBUG if i % 5 else INFO, "learning", sample_id,
                                output, Y[sample_id])
            i += 1

    # learnings_samples, test_samples: attributes => targets
    # learning_rates: epoch => new learning rate
    def work(self, epoch, learning_attributes, learning_targets,
             test_attributes, test_targets, learning_rates, it_on_last_epoch):
        for i in range(0, epoch):
            if i in learning_rates:
                learning_rate = learning_rates[i]
            # learning
            X = list(learning_attributes.items())
            Y = list(learning_targets.items())
            if (i == epoch - 1) & (0 != it_on_last_epoch):
                X = X[:it_on_last_epoch]
                Y = Y[:it_on_last_epoch]
            self.learn(X, Y, learning_rate)

            # testing
            self.predict(list(test_attributes.items()), list(test_targets.items()))

    def predict(self, X, Y):
        i = 0
        for sample_id, x in X:
            _, a = self.forward_prop(x)
            self.log_it_results(DEBUG if i % 5 else INFO, "predicting", sample_id,
                                a[self.layers - 1], Y[sample_id])
            i += 1

    def log_it_results(self, level, action_name, sample_id, output, y):
        cost = self.cost(output, y)
        cost_component = self.cost_component(output, y)
        getLogger(__name__).log(level=level, msg=("%s %f...", action_name, sample_id))
        getLogger(__name__).log(level=level, msg="y = [{}]".format(' '.join(map(str, y))))
        getLogger(__name__).log(level=level, msg="output = [{}]".format(' '.join(map(str, y))))
        getLogger(__name__).log(level=level, msg="cost: %f [{}]"
                                .format(cost, ' '.join(map(str, cost_component))))

    def log_weights_biases(self):
        getLogger(__name__).debug("weights:\n" + format_matrix_array(self.W))
        getLogger(__name__).debug("biases:\n" + format_matrix_array(self.b))
