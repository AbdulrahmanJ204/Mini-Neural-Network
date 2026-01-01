from typing import List
import numpy as np

from layers.affine import Affine
from layers.layer import Layer
from layers.normalization.batch_normalization import BatchNormalization
from layers.regularization.dropout import Dropout
from layers.loss.loss import Loss


class NeuralNetwork:
    def __init__(self, layers: List[Layer], loss_layer: Loss):
        # Layer.reset_counter()
        self.layers = layers
        self.last_layer = loss_layer
        self.initialized = False

    def add_layer(self, layer: Layer):
        """
        Adds layer to the network
        make sure to initialize the network after adding new layer if you initialized it before.
        """
        self.layers.append(layer)
        self.initialized = False

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
            if isinstance(layer, (BatchNormalization, Dropout)):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x, False)

        if y.shape[1] == 1:  # for binary classification
            y = (y >= 0).astype(int).flatten()
            t = t.flatten().astype(int)
        else:  # multi class
            y = np.argmax(y, axis=1)
            if t.ndim != 1:
                t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / len(x)
        return accuracy

    def gradient(self, x, t):
        """
        takes X (input data) and t (labels data)
        return the gradients of the network
        """
        loss = self.loss(x, t)
        dout = self.last_layer.backward()
        layers = reversed(self.layers)
        grads = []
        for layer in layers:
            dout = layer.backward(dout)
            grads.append(layer.grads())

        return list(reversed(grads))

    def structure(self):
        """
        Get network structure.
        """
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
                layer_info["params"] = (
                    layer.input_size * layer.output_size + layer.output_size
                )
                current_size = layer.output_size

            elif isinstance(layer, BatchNormalization):
                layer_info["input_size"] = current_size
                layer_info["output_size"] = current_size
                layer_info["params"] = current_size * 2  # gamma and beta

            elif isinstance(layer, Dropout):
                layer_info["dropout_rate"] = layer.dropout_ratio
                layer_info["input_size"] = current_size
                layer_info["output_size"] = current_size
                layer_info["params"] = 0

            else:
                # Activation layers
                layer_info["input_size"] = current_size
                layer_info["output_size"] = current_size
                layer_info["params"] = 0

            structure.append(layer_info)

        return _format_as_table(structure)


def _format_as_table(structure):
    """Format structure as ASCII table"""
    lines = []
    layers_len = 20
    type_len = 30
    input_len = 10
    output_len = 10
    params_len = 10
    total_len = (layers_len + type_len + input_len + output_len + params_len) + 3
    lines.append("=" * total_len)
    lines.append(
        f"{'Layer':<{layers_len}} {'Type':<{type_len}} {'Input':<{input_len}} {'Output':<{output_len}} {'Params':<{params_len}}"
    )
    lines.append("=" * total_len)

    total_params = 0
    for i, layer in enumerate(structure):
        layer_name = f"Layer {i + 1}"
        layer_type = layer["type"]
        input_size = layer.get("input_size", "-")
        output_size = layer.get("output_size", "-")
        params = layer.get("params", 0)
        total_params += params

        # Add extra info for specific layers
        extra = ""
        if layer["type"] == "Affine":
            extra = f" ({layer.get('initializer', '')})"
        elif layer["type"] == "Dropout":
            extra = f" (rate={layer.get('dropout_rate', 0):.2f})"

        lines.append(
            f"{layer_name:<{layers_len}} {layer_type + extra:<{type_len}} "
            f"{str(input_size):<{input_len}} {str(output_size):<{output_len}} {str(params):<{params_len}}"
        )

    lines.append("=" * total_len)
    lines.append(
        f"{'Total Parameters:':<{total_len - params_len}} {total_params:<{params_len}}"
    )
    lines.append("=" * total_len)

    return "\n".join(lines)
