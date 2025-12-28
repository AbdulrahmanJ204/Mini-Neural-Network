from optimizers.optimizer import Optimizer
import numpy as np


class AdaGrad(Optimizer):
    """AdaGrad (Adaptive Gradient) optimizer.

    Adapts learning rate for each parameter based on the historical sum of squared gradients.
    Parameters with larger gradients get smaller effective learning rates.
    """

    def __init__(self, lr=0.01):
        """Initialize AdaGrad optimizer.

        Args:
            lr: Initial learning rate (default: 0.01).
        """
        self.lr = lr
        self.h = {}

    def update_params(self, params, grads):
        """Update parameters using AdaGrad rule.

        Maintains sum of squared gradients (h) and adapts learning rate: lr_t = lr / sqrt(h + eps)

        Args:
            params: Dictionary of parameters to update.
            grads: Dictionary of gradients.
        """
        for key, val in params.items():
            if key not in self.h:
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
