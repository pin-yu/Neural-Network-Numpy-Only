import unittest
import numpy as np

from src.utility import one_hot_encode, shuffle, get_batch


class TestUtility(unittest.TestCase):
    def test_one_hot_encode(self):
        y = [9]

        actual = one_hot_encode(y, class_num=10)
        expected = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        assert (actual == expected).all()

    def test_shuffle(self):
        X = np.arange(30)
        y = np.arange(30)

        X_shuffle, y_shuffle = shuffle(X, y)
        assert X_shuffle.shape[0] == X.shape[0]
        assert y_shuffle.shape[0] == y.shape[0]
        assert (X_shuffle == y_shuffle).all()

    def test_batch_32(self):
        batch_size = 32
        total_data_num = 1000
        last_batch = total_data_num // batch_size

        X = np.arange(total_data_num)

        batch1 = get_batch(X, 0, batch_size, total_data_num)
        batch_last = get_batch(X, last_batch, batch_size, total_data_num)

        assert (batch1 == np.arange(32)).all()
        assert (batch_last == np.arange(992, 1000)).all()

    def test_batch_1(self):
        batch_size = 1
        total_data_num = 1000
        last_batch = total_data_num // batch_size

        X = np.arange(total_data_num)

        batch1 = get_batch(X, 0, batch_size, total_data_num)
        batch_last = get_batch(X, last_batch, batch_size, total_data_num)

        assert (batch1 == np.arange(1)).all()
        assert (batch_last == np.arange(999, 1000)).all()
