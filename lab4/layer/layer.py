import abc

class Layer(abc.ABC):

    @abc.abstractclassmethod
    # x - input
    def forward_prop(self, x):
        pass

    @abc.abstractclassmethod
    def change_weights_biases(self, learning_rate, db, dW):
        pass