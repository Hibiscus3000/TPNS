import abc

class NNPart(abc.ABC):

    def __init__(self, layer, activation_function):
        self.layer = layer
        self.activation_function = activation_function
    
    # z - layer output
    def forward_prop(self, x):
        self.x = x
        self.z = self.layer.forward_prop(x)
        if self.activation_function is not None:
            return self.activation_function.apply(self.z)
        else:
            return self.z
    
    @abc.abstractclassmethod
    # d - is expected result for output layer and gradient from next layer for all the other layers
    def back_prop(self, d):
        pass

    def change_weights_biases(self, learning_rate, db, dW):
        self.layer.change_weights_biases(learning_rate, db, dW)