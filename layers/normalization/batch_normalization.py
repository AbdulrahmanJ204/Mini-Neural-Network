import numpy as np

from layers.layer import Layer


class BatchNormalization(Layer):
    """Batch Normalization layer.

    Normalizes activations of the previous layer for each batch during training,
    which can lead to faster convergence and allows for higher learning rates.
    Uses running statistics for inference.
    """

    def __init__(self, gamma=1.0, beta=0.0, momentum=0.9):
        """Initialize Batch Normalization layer.

        Args:
            gamma: Initial scale parameter (default: 1.0).
            beta: Initial shift parameter (default: 0.0).
            momentum: Momentum for running mean/variance (default: 0.9).
        """
        self.gamma_init = gamma
        self.beta_init = beta
        self.running_mean = None
        self.running_var = None
        self.momentum = momentum
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
        self.params = {}
        super().__init__()

    def init_weights(self, input_size):
        """Initialize gamma and beta parameters.

        Args:
            input_size: Number of features.
        """
        self.params[f"gamma{self.id}"] = np.ones(input_size) * self.gamma_init
        self.params[f"beta{self.id}"] = np.ones(input_size) * self.beta_init

        self.running_mean = np.zeros(input_size)
        self.running_var = np.zeros(input_size)

    def forward(self, x, train_flg=True):
        """Forward pass through batch normalization.

        During training: Normalizes using batch statistics.
        During inference: Normalizes using running statistics.

        Args:
            x: Input of shape (batch_size, features).
            train_flg: Whether in training mode (default: True).

        Returns:
            Normalized output of same shape as input.
        """
        if self.running_mean is None:
            n, d = x.shape
            self.running_mean = np.zeros(d)
            self.running_var = np.zeros(d)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 1e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 1e-7))

        gamma = self.params[f"gamma{self.id}"]
        beta = self.params[f"beta{self.id}"]
        out = gamma * xn + beta
        return out

    def backward(self, dout):
        """Backward pass for batch normalization gradient computation.

        Args:
            dout: Gradient from next layer.

        Returns:
            Gradient with respect to input.
        """
        gamma = self.params[f"gamma{self.id}"]

        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx

    def grads(self):
        """Get parameter gradients.

        Returns:
            Dictionary containing gamma and beta gradients.
        """
        return {f"gamma{self.id}": self.dgamma, f"beta{self.id}": self.dbeta}

    def parameters(self):
        """Get layer parameters.

        Returns:
            Dictionary containing gamma and beta parameters.
        """
        return self.params
