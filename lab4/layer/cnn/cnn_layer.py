import abc

class CNNLayer(abc.ABC):

    # height, width - size of the kernel; s1, s2 - strides
    def __init__(self, size, image_depth):
        self.size = size
        self.image_depth = image_depth

    # next_d - gradint from next layer
    @abc.abstractclassmethod
    def back_prop(self, next_d):
        pass