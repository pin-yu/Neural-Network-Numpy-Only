import unittest
import numpy as np

from src.activation import Relu
from src.activation import Sigmoid
from src.activation import Softmax


class TestActivation(unittest.TestCase):
    def setUp(self):
        self.X = np.asarray([[0.1, 0, -0.1]])

    def test_relu_forward_pass(self):
        actual = Relu.forward(self.X)
        expected = np.asarray([[0.1, 0, 0]])

        assert (actual == expected).all()

    def test_relu_backward_pass(self):
        actual = Relu.backward(self.X)
        expected = np.asarray([[1, 1, 0]])

        assert (actual == expected).all()

    def test_sigmoid_forward_pass(self):
        actual = Sigmoid.forward(self.X)

        assert actual[0][0] > 0.5
        assert actual[0][1] == 0.5
        assert actual[0][2] < 0.5

    def test_sigmoid_backward_pass(self):
        actual = Sigmoid.backward(self.X)

        assert actual[0][0] < 0.25
        assert actual[0][1] == 0.25
        assert actual[0][2] < 0.25

    def test_softmax_forward(self):
        X = np.asarray([[1000, 2000, 3000]])

        actual = Softmax.forward(X)

        assert (actual >= 0).all()
        assert (actual <= 1).all()
        assert np.isclose(np.sum(actual), [1])

    def test_softmax_backward(self):
        out = np.asarray([[10, 20, 30]])
        y = np.asarray([[10, 15, 20]])

        actual = Softmax.backward(out, y)
        expected = np.asarray([[0, 5, 10]])

        assert (actual == expected).all()
