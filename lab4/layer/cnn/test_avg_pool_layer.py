import unittest
import numpy as np

from layer.cnn.avg_pool_layer import *


class TestAvgPoolLayer(unittest.TestCase):

    def test_forward_prop(self):
        ap = AvgPoolLayer(1,4,2,2)
        X = np.empty((1,4,4))
        X.fill(4)
        X[0,1:3,:].fill(8)
        self.assertTrue(np.equal(ap.forward_prop(X),np.array([[[6, 6], [6, 6]]])).all())

    def test_back_prop(self):
        ap = AvgPoolLayer(2, 8, 2, 2)
        next_d = np.empty((2,4,4))
        next_d.fill(4)
        next_d[:,1:3,:].fill(8)
        expected = np.array([[[1,1,1,1,1,1,1,1],
                             [1,1,1,1,1,1,1,1],
                             [2,2,2,2,2,2,2,2],
                             [2,2,2,2,2,2,2,2],
                             [2,2,2,2,2,2,2,2],
                             [2,2,2,2,2,2,2,2],
                             [1,1,1,1,1,1,1,1],
                             [1,1,1,1,1,1,1,1]]])
        expected = np.concatenate((expected, expected))
        d, _, _ = ap.back_prop(next_d)
        self.assertTrue(np.equal(d,expected).all())