import numpy as np
from logging import *
from formater import *


class Perceptron:
    def __init__(self, neurons):
        self.layers = len(neurons) - 1  # number of layers excluding input layer
        self.b = []
        self.W = []
        for i in range(0, self.layers):
            self.W.append(np.random.rand(neurons[i + 1], neurons[i]) - 0.5)
            self.b.append(np.random.rand(neurons[i + 1]) - 0.5)
        getLogger(__name__).debug("perceptron initialized")
        self.log_weights_biases()

    # z = Wa + b
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return np.multiply(self.sigmoid(z), 1 - self.sigmoid(z))

    # x - attributes
    def forward_prop(self, x):
        a = [x]
        z = [x]
        for l in range(0, self.layers):
            z.append(np.matmul(self.W[l], a[l]) + self.b[l])
            a.append(self.sigmoid(z[l + 1]))
        return z, a

    # output - output layer results
    def cost(self, output, y):
        C = 0  # C - cost function
        for i in range(0, len(y)):
            if y[i] is not None:
                C += (output[i] - y[i]) ** 2
        return C

    def cost_component(self, output, y):
        return [(output[i] - y[i]) ** 2 if y[i] is not None else None for i in
                range(0, len(y))]

    # z = Wa(L - 1) + b,a(L) = f(z), f - activation function, y - targets
    def back_prop(self, z, a, y):
        dW = []
        db = []

        # deltas for the output layer
        y_no_nones = [y[i] if y[i] is not None else a[self.layers][i] for i in
                      range(0, len(y))]
        db.append(self.sigmoid_derivative(z[self.layers]) * (a[self.layers] - y_no_nones))

        # deltas for hidden layers
        for i in range(self.layers - 1, 0, -1):
            db.insert(0, np.multiply(self.sigmoid_derivative(z[i]), np.matmul(db[0], self.W[i])))

        for i in range(0, self.layers):
            dW.append(np.matmul(np.transpose(np.atleast_2d(db[i])), np.atleast_2d(a[i])))

        return dW, db

    def learn_it(self, x, y, learning_rate):
        z, a = self.forward_prop(x)

        dW, db = self.back_prop(z, a, y)
        for i in range(0, self.layers):
            self.W[i] -= learning_rate * dW[i]
            self.b[i] -= learning_rate * db[i]
        return a[self.layers]

    def learn(self, X, Y, learning_rate):
        i = 0
        costs = []
        for sample_id, x in X:
            output = self.learn_it(x, Y[sample_id], learning_rate)
            costs.append(self.log_it_results(DEBUG if i % 5 else INFO, "learning", sample_id,
                                             output, Y[sample_id]))
            i += 1

        return costs

    # learnings_samples, test_samples: attributes => targets
    # learning_rates: epoch => new learning rate
    def learn_and_predict(self, epoch, learning_attributes, learning_targets,
                          test_attributes, test_targets, learning_rates, it_on_last_epoch, critical_cost):
        self.critical_cost = critical_cost
        costs_learning = []
        costs_testing = []
        for i in range(0, epoch):
            if i in learning_rates:
                learning_rate = learning_rates[i]
            # learning
            X = list(learning_attributes.items())
            Y = list(learning_targets.items())
            if (i == epoch - 1) & (0 != it_on_last_epoch):
                X = X[:it_on_last_epoch]
                Y = Y[:it_on_last_epoch]
            Y = {sample_id: sample for sample_id, sample in Y}
            costs_learning += self.learn(X, Y, learning_rate)

            # testing
            costs_testing += self.predict(list(test_attributes.items()), test_targets)

        return costs_learning, costs_testing

    def predict(self, X, Y):
        i = 0
        costs = []
        for sample_id, x in X:
            _, a = self.forward_prop(x)
            costs.append(self.log_it_results(DEBUG if i % 5 else INFO, "predicting", sample_id,
                                             a[self.layers], Y[sample_id]))
            i += 1

        return costs

    def log_it_results(self, level, action_name, sample_id, output, y):
        getLogger(__name__).log(level=level, msg="{} {}...".format(action_name, sample_id))
        cost = self.cost(output, y)
        cost_component = self.cost_component(output, y)
        getLogger(__name__).log(level=level, msg="y = [{}]".format(' '.join(map(str, y))))
        getLogger(__name__).log(level=level, msg="output = [{}]".format(' '.join(map(str, output))))
        getLogger(__name__).log(level=level, msg="[{}{}] cost: {} [{}]"
                                .format(action_name[0], sample_id,
                                        cost, ' '.join(map(str, cost_component))))
        for i in range(0, len(cost_component)):
            if (cost_component[i] is not None):
                if (cost_component[i] > self.critical_cost):
                    getLogger(__name__).warning("Cost function exceeds critical limit: {:s} {:d}:{:d}:{:.3f}"
                                                .format(action_name, sample_id, i, cost_component[i]))
        return cost_component

    def log_weights_biases(self):
        getLogger(__name__).debug("weights:\n" + format_matrix_array(self.W))
        getLogger(__name__).debug("biases:\n" + format_matrix_array(self.b))
