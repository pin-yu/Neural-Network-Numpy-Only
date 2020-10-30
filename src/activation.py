'''
This module provides common activation functions.
'''

import numpy as np

# Relu


class Relu:
    @staticmethod
    def forward(X):
        return np.maximum(0, X)

    @staticmethod
    def backward(X):
        return np.where(X < 0, 0, 1)


class Sigmoid:
    @staticmethod
    def forward(X):
        # stable version of sigmoid function
        return 1 / (np.exp(-X) + 1)

    @staticmethod
    def backward(X):
        return np.exp(-X) / (np.exp(-X) + 1) ** 2


class Softmax:
    @staticmethod
    def forward(X):
        max_X = np.max(X, axis=1, keepdims=True)
        stable_X = X - max_X
        return np.exp(stable_X) / np.sum(np.exp(stable_X),
                                         axis=1, keepdims=True)

    @staticmethod
    def backward(cost, y):
        # Notice!
        # The following formula is the derivative of softmax with cross entropy.
        # Such a beautiful formula huh?
        return cost - y
