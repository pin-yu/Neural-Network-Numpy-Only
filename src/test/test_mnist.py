import unittest

from src.mnist import Mnist


class TestActivation(unittest.TestCase):
    def test_mnist_get(self):
        X, y = Mnist().get()

        assert (X.shape == (70000, 784))
        assert (y.shape == (70000,))
        assert (isinstance(y[0], str))

    def test_mnist_normalization(self):
        X, _ = Mnist().normalization().get()

        assert (X[0] <= 1.).all()
        assert (X[0] >= 0.).all()

    def test_mnist_one_hot_encode(self):
        _, y = Mnist().one_hot_encode().get()

        assert (y <= 9).all()
        assert (y >= 0).all()
