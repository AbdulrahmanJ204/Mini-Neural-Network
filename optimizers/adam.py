from optimizers.optimizer import Optimizer
import numpy as np


class Adam(Optimizer):
    """Adam (Adaptive Moment Estimation) optimizer.

    Combines momentum and RMSprop by maintaining exponential moving averages
    of both gradients (first moment) and squared gradients (second moment).
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        """Initialize Adam optimizer.

        Args:
            lr: Learning rate (default: 0.001).
            beta1: Exponential decay rate for first moment (default: 0.9).
            beta2: Exponential decay rate for second moment (default: 0.999).
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = {}
        self.v = {}

    def update_params(self, params, grads):
        """Update parameters using Adam rule.

        Maintains first moment (m) and second moment (v) estimates with bias correction.

        Args:
            params: Dictionary of parameters to update.
            grads: Dictionary of gradients.
        """
        for key, val in params.items():
            if key not in self.m:
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = (
            self.lr
            * np.sqrt(1.0 - self.beta2**self.iter)
            / (1.0 - self.beta1**self.iter)
        )

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
