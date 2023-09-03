import abc

class Cell(abc.ABC):

    def __init__(self, window_size):
        self.window_size = window_size

    @abc.abstractclassmethod
    def train_all(self, X, Y):
        pass

    @abc.abstractclassmethod
    def test_all(self, X):
        pass

    @abc.abstractclassmethod
    # X - input
    def forward_prop(self, X):
        pass

    @abc.abstractclassmethod
    # Y - expected
    def back_prop(self, Y):
        pass