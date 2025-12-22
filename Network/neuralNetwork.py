from typing import List
import numpy as np

from layers.layer import Affine, Layer
from layers.optimization.BatchNormalization import BatchNormalization


class NeuralNetwork:
    def __init__(self, layers: List[Layer], lastLayer: Layer):
        self.layers = layers
        self.lastLayer = lastLayer
        self.initalized = False

    def init_weights(self, input_size):
        current_size = input_size
        self.initalized = True
        for layer in self.layers:
            if isinstance(layer, Affine):
                layer.init_weights(current_size)
                current_size = layer.output_size
            elif isinstance(layer, BatchNormalization):
                # BatchNormalization doesn't change dimensions
                layer.init_weights(current_size)

    def predict(self, x, train_flg=True):
        for layer in self.layers:
            if isinstance(layer, BatchNormalization):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x, False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            """
              for one hot encoding i think
            [[0, 0, 1],    # Sample 0: class 2
              [1, 0, 0],    # Sample 1: class 0
              [0, 1, 0]]    # Sample 2: class 1
            convert it to : [
              2 ,
              0 ,
              1
            ]
            """
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / len(x)
        return accuracy

    def gradient(self, x, t):
        loss = self.loss(x, t)
        dout = self.lastLayer.backward()
        layers = reversed(self.layers)
        grads = []
        for layer in layers:
            dout = layer.backward(dout)
            grads.append(layer.grads())

        return list(reversed(grads))
