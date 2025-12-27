import numpy as np

from layers.layer import Layer



class BatchNormalization(Layer):
    def __init__(self, gamma=1.0, beta=0.0, momentum=0.9):
        self.gamma_init = gamma
        self.beta_init = beta
        self.running_mean = None
        self.running_var = None
        self.momentum = momentum
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
        self.params = {}
        super().__init__()

    def init_weights(self, input_size):
        self.params[f"gamma{self.cnt}"] = np.ones(input_size) * self.gamma_init
        self.params[f"beta{self.cnt}"] = np.ones(input_size) * self.beta_init

        self.running_mean = np.zeros(input_size)
        self.running_var = np.zeros(input_size)

    def forward(self, x, train_flg=True):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

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
            xn = xc / ((np.sqrt(self.running_var + 1e-7)))

        gamma = self.params[f"gamma{self.cnt}"]
        beta = self.params[f"beta{self.cnt}"]
        out = gamma * xn + beta
        return out

    def backward(self, dout):
        gamma = self.params[f"gamma{self.cnt}"]
        beta = self.params[f"beta{self.cnt}"]

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
        return {f"gamma{self.cnt}": self.dgamma, f"beta{self.cnt}": self.dbeta}

    def parameters(self):
        return self.params
