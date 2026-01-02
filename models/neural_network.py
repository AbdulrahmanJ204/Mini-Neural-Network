from typing import List
import numpy as np

from layers.affine import Affine
from layers.layer import Layer
from layers.normalization.batch_normalization import BatchNormalization
from layers.regularization.dropout import Dropout
from layers.loss.loss import Loss


class NeuralNetwork:
    """Neural network model combining multiple layers.

    Manages the composition of layers, weight initialization, forward/backward passes,
    and provides utility methods for evaluation and inspection.
    """

    def __init__(self, layers: List[Layer], loss_layer: Loss):
        """Initialize neural network.

        Args:
            layers: List of Layer objects forming the network.
            loss_layer: Loss function layer for computing the loss.
        """
        # Layer.reset_counter()
        self.layers = layers
        self.last_layer = loss_layer
        self.initialized = False

    def add_layer(self, layer: Layer):
        """Add a new layer to the network.

        Note: Make sure to reinitialize the network after adding new layers if it was
        previously initialized.

        Args:
            layer: Layer object to add to the network.
        """
        self.layers.append(layer)
        self.initialized = False

    def init_weights(self, input_size):
        """Initialize weights for all layers based on input size.

        Sequentially initializes weights for Affine and BatchNormalization layers,
        tracking the output size of each layer to determine the input size of the next.

        Args:
            input_size: Number of input features.
        """
        current_size = input_size
        self.initialized = True
        for layer in self.layers:
            if isinstance(layer, Affine):
                layer.init_weights(current_size)
                current_size = layer.output_size
            elif isinstance(layer, BatchNormalization):
                layer.init_weights(current_size)

    def predict(self, x, train_flg=True):
        """Forward pass through the network to get predictions.

        Args:
            x: Input data of shape (batch_size, input_size).
            train_flg: Whether in training mode (affects Dropout and BatchNormalization).

        Returns:
            Network predictions (output of last layer before loss).
        """
        for layer in self.layers:
            if isinstance(layer, (BatchNormalization, Dropout)):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        """Compute loss on input-target pair.

        Args:
            x: Input data of shape (batch_size, input_size).
            t: Target labels of shape (batch_size, num_classes).

        Returns:
            Loss value (scalar).
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        """Compute classification accuracy on input-target pair.

        Args:
            x: Input data of shape (batch_size, input_size).
            t: Target labels (one-hot encoded or class indices).

        Returns:
            Accuracy value in range [0, 1].
        """
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
        """Compute gradients of all network parameters using backpropagation.

        Args:
            x: Input data of shape (batch_size, input_size).
            t: Target labels of shape (batch_size, num_classes).

        Returns:
            List of gradient dictionaries, one for each layer (in order).
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
        """Get network structure as formatted ASCII table.

        Returns:
            Formatted string containing network structure with layer details and parameter counts.
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
    """Format network structure as ASCII table.

    Args:
        structure: List of layer information dictionaries.

    Returns:
        Formatted string representation of the network structure.
    """
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
