from typing import List
import numpy as np

from layers.affine import Affine
from layers.layer import Layer
from layers.normalization.batch_normalization import BatchNormalization
from layers.regularization.dropout import Dropout


class NeuralNetwork:
    def __init__(self, layers: List[Layer], last_layer: Layer):
        Layer.reset_counter()
        self.layers = layers
        self.last_layer = last_layer
        self.initialized = False

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def init_weights(self, input_size):
        current_size = input_size
        self.initialized = True
        for layer in self.layers:
            if isinstance(layer, Affine):
                layer.init_weights(current_size)
                current_size = layer.output_size
            elif isinstance(layer, BatchNormalization):
                layer.init_weights(current_size)

    def predict(self, x, train_flg=True):
        for layer in self.layers:
            if isinstance(layer, BatchNormalization) or isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x, False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / len(x)
        return accuracy

    def gradient(self, x, t):
        loss = self.loss(x, t)
        dout = self.last_layer.backward()
        layers = reversed(self.layers)
        grads = []
        for layer in layers:
            dout = layer.backward(dout)
            grads.append(layer.grads())

        return list(reversed(grads))

    def structure(self):
        structure = []
        current_size = None

        for layer in self.layers:
            layer_info = {
                "type": type(layer).__name__,
                "output_size": 0,
                "input_size": 0,
            }

            if isinstance(layer, Affine):
                layer_info["input_size"] = layer.input_size
                layer_info["output_size"] = layer.output_size
                layer_info["initializer"] = type(layer.initializer).__name__
                current_size = layer.output_size

            elif isinstance(layer, BatchNormalization):
                layer_info["input_size"] = current_size

            elif isinstance(layer, Dropout):
                layer_info["dropout_rate"] = layer.dropout_ratio

            structure.append(layer_info)

        return structure
