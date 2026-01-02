from typing import Type

from layers import Loss
from tuning.tuner import Tuner
from layers.affine import Affine
from layers.normalization.batch_normalization import BatchNormalization
from layers.regularization.dropout import Dropout
from models.neural_network import NeuralNetwork
from training.trainer import Trainer
from optimizers.adam import Adam
from optimizers.momentum import Momentum

import copy
from itertools import product


class LayerSpecs:
    """Specification for a hidden layer configuration.

    Encapsulates the parameters needed to build a layer including
    neurons, initialization, activation, dropout, and batch normalization.
    """

    def __init__(self, params):
        """Initialize layer specification.

        Args:
            params: Dictionary containing layer configuration parameters:
                - layer_neurons_number: Number of neurons.
                - init_method: Weight initializer class.
                - activation: Activation function class.
                - dropout_rate: Dropout probability.
                - batch_normalization: Whether to use batch norm.
        """
        self.neurons_number = params["layer_neurons_number"]
        self.init_method = params["init_method"]
        self.activation = params["activation"]
        self.dropout_rate = params["dropout_rate"]
        self.batch_normalization = params["batch_normalization"]

    def build(self, is_last_hidden=False):
        """Build the layer sequence from specification.

        Args:
            is_last_hidden: Whether this is the last hidden layer (affects dropout placement).

        Returns:
            List of Layer objects implementing the specification.
        """
        layers = [
            Affine(self.neurons_number, self.init_method()),
        ]

        if self.batch_normalization:
            layers.append(BatchNormalization())

        layers.append(self.activation())

        if not is_last_hidden:
            layers.append(Dropout(self.dropout_rate))

        return layers


class GridTuner(Tuner):
    """Grid search hyperparameter tuning.

    Exhaustively searches over all combinations of specified hyperparameters
    by evaluating every combination in a grid.
    """

    def __init__(self):
        """Initialize grid tuner."""
        self.best_params = None
        self._networks_config = []
        self._possible_layers = []
        self._possible_optimizers = []
        self._hidden_layers = []
        self._layer_params_keys = [
            "layer_neurons_number",
            "init_method",
            "activation",
            "dropout_rate",
            "batch_normalization",
        ]
        self._params = {}
        super().__init__()


    def reset(self):
        """Reset internal state for a new search."""
        self._networks_config = []
        self._possible_layers = []
        self._possible_optimizers = []
        self._hidden_layers = []

    def __generate_layer(self, params):
        """Generate a layer specification from parameter values.

        Args:
            params: Dictionary with layer parameters.
        """
        self._possible_layers.append(LayerSpecs(params))

    def __generate_possible_layers(self):
        """Generate all possible layer configurations by combinatorial product."""
        param_lists = [
            self._params["layer_props"][key] for key in self._layer_params_keys
        ]

        for combo in product(*param_lists):
            layer_params = dict(zip(self._layer_params_keys, combo))
            self.__generate_layer(layer_params)

    def __generate_possible_optimizers(self):
        """Generate all possible optimizer configurations."""
        for optimizer in self._params["optimizer"]:
            for learning_rate in self._params["learning_rate"]:
                if optimizer == Adam:
                    for beta1 in self._params["beta1"]:
                        for beta2 in self._params["beta2"]:
                            self._possible_optimizers.append(
                                optimizer(lr=learning_rate, beta1=beta1, beta2=beta2)
                            )

                elif optimizer == Momentum:
                    for momentum in self._params["momentum"]:
                        self._possible_optimizers.append(
                            optimizer(momentum=momentum, lr=learning_rate)
                        )

                else:
                    self._possible_optimizers.append(optimizer(lr=learning_rate))

    def __generate_hidden_layers(self, n_layers):
        """Generate all combinations of n hidden layers.

        Args:
            n_layers: Number of hidden layers.
        """
        for combo in product(self._possible_layers, repeat=n_layers):
            self._hidden_layers.append(
                [
                    layer.build(is_last_hidden=(i == n_layers - 1))
                    for i, layer in enumerate(combo)
                ]
            )

    def __generate_possible_networks(self, output_layer, loss_layer_cls: Type[Loss]):
        """Generate all possible network architectures.

        Args:
            output_layer: Output layer specification.
            loss_layer_cls: Loss function class.
        """
        for n_layers in self._params["hidden_number"]:
            self._hidden_layers = []
            self.__generate_hidden_layers(n_layers)
            for hidden_set in self._hidden_layers:
                layers = [layer for collect in hidden_set for layer in collect]
                layers.append(copy.deepcopy(output_layer))

                self._networks_config.append({"layers": layers, "loss": loss_layer_cls})

    def optimize(
        self,
        params: dict,
        x_train,
        x_val,
        t_train,
        t_val,
        output_layer: Affine,
        loss_layer_cls: Type[Loss],
    ):
        """Perform grid search over all hyperparameter combinations.

        Args:
            params: Dictionary of hyperparameter ranges to search over.
            x_train: Training input data.
            x_val: Validation input data.
            t_train: Training target labels.
            t_val: Validation target labels.
            output_layer: Output layer specification.
            loss_layer_cls: Loss function class.

        Returns:
            Dictionary of best found hyperparameters.
        """
        self.reset()
        self._params = params
        self.__generate_possible_layers()
        self.__generate_possible_optimizers()
        self.__generate_possible_networks(output_layer, loss_layer_cls)
        best_accuracy = 0
        self.best_params = {}
        for optimizer_o in self._possible_optimizers:
            for network_config in self._networks_config:
                for batch_size in self._params["batch_size"]:
                    for epoch in self._params["epochs"]:

                        network = NeuralNetwork(
                            network_config["layers"], network_config["loss"]()
                        )
                        network.init_weights(x_train.shape[1])
                        optimizer = copy.deepcopy(optimizer_o)
                        trainer = Trainer(network, optimizer)
                        loss, acc = trainer.fit(
                            x_train, x_val, t_train, t_val, batch_size, epoch
                        )
                        accuracy = trainer.evaluate(x_val, t_val)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            self.best_trainer = trainer
                            self.best_params = {
                                "optimizer": type(optimizer).__name__,
                                "learning_rate": optimizer.lr,
                                "beta1": (
                                    0
                                    if not isinstance(optimizer, Adam)
                                    else optimizer.beta1
                                ),
                                "beta2": (
                                    0
                                    if not isinstance(optimizer, Adam)
                                    else optimizer.beta2
                                ),
                                "momentum": (
                                    0
                                    if not isinstance(optimizer, Momentum)
                                    else optimizer.momentum
                                ),
                                "batch_size": batch_size,
                                "epochs": epoch,
                            }
        return self.best_params
