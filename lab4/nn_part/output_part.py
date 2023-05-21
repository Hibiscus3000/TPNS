from nn_part.nn_part import *

class OutputPart(NNPart):

    # d - is the expected result
    def back_prop(self, d):
        # assert self.z is not None
        return self.layer.back_prop(self.x, self.z, self.activation_function, d)