import abc

class Coder(abc.ABC):

    @abc.abstractmethod
    def encode(self, samples, withNones):
        """encode string attributes into numbers"""

    @abc.abstractmethod
    def decode(self, deltas, results):
        """decode perceptron results"""
