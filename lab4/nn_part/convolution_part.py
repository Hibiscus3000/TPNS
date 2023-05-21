import numpy as np

from nn_part.nn_part import *

class ConvolutionPart(NNPart):

    def back_prop(self,d):
        # assert self.x is not None
        # assert self.z is not None
        return self.layer.back_prop(self.x, d * self.activation_function.derivative(self.z))