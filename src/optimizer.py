import numpy as np


class Optimizer:
    def __init__(self, optimizer='momentum', lr_rate=.1, beta=.9):
        self.optimizer = optimizer
        self.lr_rate = lr_rate
        self.beta = beta

        self.moving_average = {}

    def update(self, nn):
        if not bool(self.moving_average):
            self._init_moving_average(nn)

        if self.optimizer == 'momentum':
            self._momentum_update(nn)

        elif self.optimizer == 'sgd':
            self._sgd_update(nn)

        else:
            raise ValueError(
                "We don't support {} optimizer.".format(optimizer))

    def _momentum_update(self, nn):
        for param in nn.params:
            assert self.moving_average[param].shape == nn.grads[param].shape
            # The direction of moving average and gradient should be different.
            self.moving_average[param] = self.beta * \
                self.moving_average[param] - (1. - self.beta) * nn.grads[param]
            nn.params[param] += self.lr_rate * self.moving_average[param]

    def _sgd_update(self, nn):
        for param in nn.params:
            nn.params[param] -= self.lr_rate * nn.grads[param]

    def _init_moving_average(self, nn):
        for param in nn.params:
            self.moving_average[param] = np.zeros(nn.params[param].shape)
