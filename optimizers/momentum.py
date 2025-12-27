from optimizers.optimizer import Optimizer


import numpy as np


class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def updateParams(self, params, grads):

        for key, val in params.items():
            if key not in self.v:
                self.v[key] = np.zeros_like(val)
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
