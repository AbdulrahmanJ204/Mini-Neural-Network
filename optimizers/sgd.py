from optimizers.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def updateParams(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
