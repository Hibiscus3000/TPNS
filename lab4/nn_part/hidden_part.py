from nn_part.nn_part import *

class HiddenPart(NNPart):

    def back_prop(self, d):
        assert self.z is not None
        return self.layer.back_prop(self.x, self.z, self.activation_function, d)

