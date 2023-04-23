from logging import *

from formater import *


class Perceptron:
    def __init__(self, neurons, coder, need_to_decode):
        self.layers = len(neurons) - 1  # number of layers excluding input layer
        self.coder = coder
        self.need_to_decode = need_to_decode
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
        return db, dW, a[self.layers]

    def learn(self, X, Y, learning_rate):
        i = 0
        dbs = []
        dWs = []
        result = []
        for sample_id, x in X:
            db, dW, output = self.learn_it(x, Y[sample_id], learning_rate)
            dbs.append(db)
            dWs.append(dW)
            result.append(output)
            i += 1

        return dbs, dWs, result

    def predict(self, X):
        i = 0
        result = {}
        for sample_id, x in X:
            _, a = self.forward_prop(x)
            i += 1
            result[sample_id] = a[self.layers]
        return result

    def undo_learning(self, db, dW, learning_rate):
        for j in range(0, len(dW)):
            for i in range(0, self.layers):
                self.W[i] += learning_rate * dW[j][i]
                self.b[i] += learning_rate * db[j][i]

    # learnings_samples, test_samples: attributes => targets
    # learning_rates: epoch => new learning rate
    def learn_and_predict(self, epoch, learning_attributes, learning_targets,
                          test_attributes, test_targets, learning_rates, it_on_last_epoch, meter):
        dbs, dWs = [], []
        i = 0
        for i in range(0, epoch):
            getLogger(__name__).info(f'epoch {i}...')
            if i in learning_rates:
                learning_rate = learning_rates[i]
            # learning
            X = list(learning_attributes.items())
            Y = list(learning_targets.items())
            if (i == epoch - 1) & (0 != it_on_last_epoch):
                X = X[:it_on_last_epoch]
                Y = Y[:it_on_last_epoch]
            Y = {sample_id: sample for sample_id, sample in Y}

            # learning
            db, dW, result_learning = self.learn(X, Y, learning_rate)
            dbs += db
            dWs += dW

            # testing
            result_predicting = self.predict(list(test_attributes.items()))

            rl = np.transpose(result_learning)
            el = np.transpose(list(Y.values()))
            rp = np.transpose(list(result_predicting.values()))
            ep = np.transpose(list(test_targets.values()))
            if self.need_to_decode:
                rl = self.coder.decode_targets(rl)
                el = self.coder.decode_targets(el)
                rp = self.coder.decode_targets(rp)
                ep = self.coder.decode_targets(ep)

            deteriorations = meter.count_metrics(rl, el, rp, ep)
            if deteriorations:
                self.undo_learning(dbs[-(deteriorations * len(X)):0], dWs[-(deteriorations * len(X)):0], learning_rate)
                break

        getLogger(__name__).info(f'learned {i} epoch')


    def log_weights_biases(self):
        getLogger(__name__).debug("weights:\n" + format_matrix_array(self.W))
        getLogger(__name__).debug("biases:\n" + format_matrix_array(self.b))
