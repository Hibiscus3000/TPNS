import abc

class Coder(abc.ABC):

    @abc.abstractmethod
    def encode_attributes(self, samples):
        """encode string attributes into numbers"""

    @abc.abstractmethod
    def encode_targets(self, samples):
        """encode string attributes into numbers"""

    @abc.abstractmethod
    def decode(self, deltas, results):
        """decode perceptron results"""
