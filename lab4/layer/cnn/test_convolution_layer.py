import unittest
import numpy as np

from layer.cnn.convolution_layer import *


class TestConvolutionLayer(unittest.TestCase):

    def test_forward_prop(self):
        cl = ConvolutionLayer(1, 3, 1, 1)
        cl.W = np.array([[[[3, 3, 1], [1, 3, 3], [3, 1, 3]]]])
        cl.b = np.array([1])
        X = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        output = cl.forward_prop(X)
        self.assertTrue(np.equal(output, np.array(
            [[[29, 52, 33], [64, 106, 72], [63, 92, 69]]])).all())
        
    def test_forward_prop_2_image_depth_2_filters(self):
        cl = ConvolutionLayer(2, 3, 2, 1)
        cl.W = np.array([[[[3, 3, 1], [1, 3, 3], [3, 1, 3]],[[3, 3, 1], [1, 3, 3], [3, 1, 3]]],
                         [[[3, 3, 1], [1, 3, 3], [3, 1, 3]],[[3, 3, 1], [1, 3, 3], [3, 1, 3]]]])
        cl.b = np.array([1, 1])
        X = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        output = cl.forward_prop(X)
        self.assertTrue(np.equal(output, np.array(
            [[[57, 103, 65], [127, 211, 143], [125, 183, 137]],
            [[57, 103, 65], [127, 211, 143], [125, 183, 137]]])).all())

    def test_back_prop(self):
        cl = ConvolutionLayer(1, 3, 1, 1)
        cl.W = np.array([[[[3, 3, 1], [1, 3, 3], [3, 1, 3]]]])
        cl.b = np.array([1])
        X = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        next_d = np.array([[[3, 3, 1], [1, 3, 3], [3, 1, 3]]])
        d, db, dW = cl.back_prop(X, next_d)
        self.assertTrue(np.equal(d, np.array(
            [[[9, 6, 19, 6, 9], [18, 24, 30, 20, 6], [15, 38, 45, 30, 19],
              [6, 24, 38, 24, 6], [1, 6, 15, 18, 9]]])).all())
        self.assertTrue(np.equal(db, np.array([[np.sum(next_d)]])).all())
        self.assertTrue(np.equal(dW, np.array(
            [[[[28, 51, 32], [63, 105, 71], [62, 91, 68]]]])).all())
        
    def test_back_prop_2_image_depth_2_filters(self):
        cl = ConvolutionLayer(2, 3, 2, 1)
        cl.W = np.array([[[[3, 3, 1], [1, 3, 3], [3, 1, 3]],[[3, 3, 1], [1, 3, 3], [3, 1, 3]]],
                         [[[3, 3, 1], [1, 3, 3], [3, 1, 3]],[[3, 3, 1], [1, 3, 3], [3, 1, 3]]]])
        cl.b = np.array([1,1])
        X = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        next_d = np.array([[[3, 3, 1], [1, 3, 3], [3, 1, 3]],[[3, 3, 1], [1, 3, 3], [3, 1, 3]]])
        d, db, dW = cl.back_prop(X, next_d)
        self.assertTrue(np.equal(d, 2 * np.array(
            [[[9, 6, 19, 6, 9], [18, 24, 30, 20, 6], [15, 38, 45, 30, 19],
              [6, 24, 38, 24, 6], [1, 6, 15, 18, 9]],[[9, 6, 19, 6, 9], [18, 24, 30, 20, 6], [15, 38, 45, 30, 19],
              [6, 24, 38, 24, 6], [1, 6, 15, 18, 9]]])).all())
        self.assertTrue(np.equal(db[0], np.array([[np.sum(next_d[0])]])).all())
        self.assertTrue(np.equal(db[1], np.array([[np.sum(next_d[1])]])).all())
        self.assertTrue(np.equal(dW, np.array(
            [[[[28, 51, 32], [63, 105, 71], [62, 91, 68]]],
             [[[28, 51, 32], [63, 105, 71], [62, 91, 68]]]])).all())