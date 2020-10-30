import unittest
import numpy as np

from src.loss import cross_entropy


class TestLoss(unittest.TestCase):
    def test_small_cross_entropy(self):
        out = np.asarray([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        y = np.asarray([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        actual = cross_entropy(out, y)
        assert actual < 1e-6

    def test_small_cross_entropy(self):
        out = np.asarray([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        y = np.asarray([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

        actual = cross_entropy(out, y)
        assert actual > 13
