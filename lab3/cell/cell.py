import abc

class Cell(abc.ABC):

    @abc.abstractclassmethod
    # X - input
    def forward_prop(self, X):
        pass

    @abc.abstractclassmethod
    # Y - expected
    def back_prop(self, Y):
        pass