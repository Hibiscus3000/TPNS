import abc

class ActivationFunction:

    @abc.abstractmethod
    def apply(self, z):
        pass

    @abc.abstractmethod
    def derivative(self, z):
        pass