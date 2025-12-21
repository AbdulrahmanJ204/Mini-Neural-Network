import numpy as np


class MeanSquaredError:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, y, t):
        self.y = y
        self.t = t

        self.loss = np.sum((y - t) ** 2) / len(self.t)
        return self.loss

    def backward(self, dout=1):
        dx = dout * 2 * (self.y - self.t) / len(self.t)
        return dx


# بعض المراجع بتقسم على
# 2 * N
# بالforward
#  ساعتها بكون المشتق ما فيه 2
