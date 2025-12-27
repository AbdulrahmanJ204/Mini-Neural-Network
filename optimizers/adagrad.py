from optimizers.optimizer import Optimizer

import numpy as np


class AdaGrad(Optimizer):
    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = {}

    def updateParams(self, params, grads):

        for key, val in params.items():
            if key not in self.h:
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
