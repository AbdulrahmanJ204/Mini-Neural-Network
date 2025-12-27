from layers.initializers.initializer import Initializer
import numpy as np


class SmallGaussian(Initializer):
    def init(self, input, output):
        return np.random.randn(input, output) * 0.01
