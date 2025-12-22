from layers.initializers.initializer import Initializer


import numpy as np


class XavierNormal(Initializer):
    def init(self, input, output):
        return np.random.randn(input, output) / np.sqrt(input)
