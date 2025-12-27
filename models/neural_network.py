from typing import List
import numpy as np

from layers.affine import Affine
from layers.layer import Layer
from layers.normalization.batch_normalization import BatchNormalization
from layers.regularization.dropout import Dropout
from layers.loss.loss import Loss

# TODO : format this
class NeuralNetwork:
    def __init__(self, layers: List[Layer], loss_layer: Loss):
        # Layer.reset_counter()
        self.layers = layers
        self.last_layer = loss_layer
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


    def structure(self, format='tree'):
        """
        Get network structure.

        Args:
            format: 'dict', 'table', or 'tree'
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
                layer_info["params"] = layer.input_size * layer.output_size + layer.output_size
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

        if format == 'dict':
            return structure
        elif format == 'table':
            return self._format_as_table(structure)
        elif format == 'tree':
            return self._format_as_tree(structure)
        else:
            raise ValueError(f"Unknown format: {format}")


    def _format_as_table(self, structure):
        """Format structure as ASCII table"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"{'Layer':<20} {'Type':<20} {'Input':<10} {'Output':<10} {'Params':<10}")
        lines.append("=" * 80)

        total_params = 0
        for i, layer in enumerate(structure):
            layer_name = f"Layer {i + 1}"
            layer_type = layer['type']
            input_size = layer.get('input_size', '-')
            output_size = layer.get('output_size', '-')
            params = layer.get('params', 0)
            total_params += params

            # Add extra info for specific layers
            extra = ""
            if layer['type'] == 'Affine':
                extra = f" ({layer.get('initializer', '')})"
            elif layer['type'] == 'Dropout':
                extra = f" (rate={layer.get('dropout_rate', 0):.2f})"

            lines.append(
                f"{layer_name:<20} {layer_type + extra:<20} "
                f"{str(input_size):<10} {str(output_size):<10} {str(params):<10}"
            )

        lines.append("=" * 80)
        lines.append(f"{'Total Parameters:':<50} {total_params:<10}")
        lines.append("=" * 80)

        return "\n".join(lines)


    def _format_as_tree(self, structure):
        """Format structure as tree"""
        lines = []
        lines.append("\nNetwork Architecture:")
        lines.append("=" * 60)

        for i, layer in enumerate(structure):
            is_last = (i == len(structure) - 1)
            prefix = "└── " if is_last else "├── "

            layer_type = layer['type']
            details = []

            if layer_type == 'Affine':
                details.append(f"in={layer.get('input_size', 0)}")
                details.append(f"out={layer.get('output_size', 0)}")
                details.append(f"init={layer.get('initializer', 'N/A')}")
            elif layer_type == 'Dropout':
                details.append(f"rate={layer.get('dropout_rate', 0):.2f}")
            elif layer_type == 'BatchNormalization':
                details.append(f"size={layer.get('input_size', 0)}")

            detail_str = ", ".join(details) if details else ""
            lines.append(f"{prefix}{layer_type}" + (f" ({detail_str})" if detail_str else ""))

        lines.append("=" * 60)
        return "\n".join(lines)
