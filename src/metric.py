import numpy as np


def accuracy(output, y):
    return np.mean(np.argmax(output, axis=1) == np.argmax(y, axis=1))
