from layers.initializers.initializer import Initializer
import numpy as np


class XavierNormal(Initializer):
    def init(self, input_size, output):
        return np.random.randn(input_size, output) / np.sqrt(input_size)
