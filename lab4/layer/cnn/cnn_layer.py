import abc

class CNNLayer(abc.ABC):

    # height, width - size of the kernel; s1, s2 - strides
    def __init__(self, height, width):
        self.height = height
        self.width = width

    # next_d - gradint from next layer
    def back_prop(self, next_d):
        pass