from nn_part.nn_part import *

class PoolingPart(NNPart):

    def __init__(self, layer):
        super().__init__(layer, None)

    def back_prop(self, d):
        return self.layer.back_prop(d)