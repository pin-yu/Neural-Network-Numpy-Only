import unittest
import numpy as np

from src.metric import accuracy


class TestMetric(unittest.TestCase):
    def test_half_accuracy(self):
        out = np.asarray([
            [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0.2]])
        y = np.asarray([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

        actual = accuracy(out, y)
        assert actual == 0.5

    def test_full_accuracy(self):
        out = np.asarray([
            [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0.2]])
        y = np.asarray([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        ])

        actual = accuracy(out, y)
        assert actual == 1.

    def test_zero_accuracy(self):
        out = np.asarray([
            [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0.2]])
        y = np.asarray([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

        actual = accuracy(out, y)
        assert actual == 0.
