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

import numpy as np

import copy


def get_random_optimizer(params):
    """Create a random optimizer from specified parameter options.

    Randomly samples optimizer class, learning rate, and optimizer-specific
    hyperparameters (beta1/beta2 for Adam, momentum for Momentum).

    Args:
        params: Dictionary containing optimizer configuration options:
            - optimizer: List of optimizer classes to choose from.
            - learning_rate: List of learning rates.
            - beta1: List of beta1 values (for Adam).
            - beta2: List of beta2 values (for Adam).
            - momentum: List of momentum values (for Momentum optimizer).

    Returns:
        Instantiated optimizer object with randomly sampled parameters.
    """
    optimizer_class = np.random.choice(params["optimizer"])
    learning_rate = np.random.choice(params["learning_rate"])

    if optimizer_class == Adam:
        beta1 = np.random.choice(params["beta1"])
        beta2 = np.random.choice(params["beta2"])
        optimizer = optimizer_class(lr=learning_rate, beta1=beta1, beta2=beta2)
    elif optimizer_class == Momentum:
        momentum = np.random.choice(params["momentum"])
        optimizer = optimizer_class(momentum=momentum, lr=learning_rate)
    else:
        optimizer = optimizer_class(lr=learning_rate)
    return optimizer


def generate_random_hidden_layers(params):
    """Generate a random network architecture.

    Randomly samples the number of hidden layers and randomly configures each layer
    with different neuron counts, initializers, activations, dropout rates, and
    batch normalization settings.

    Args:
        params: Dictionary containing layer configuration options:
            - hidden_number: List of possible numbers of hidden layers.
            - layer_props: Dictionary with lists of options for:
                - layer_neurons_number
                - init_method
                - activation
                - dropout_rate
                - batch_normalization

    Returns:
        Tuple of (layers, hidden_number) where:
        - layers: List of Layer objects forming the network.
        - hidden_number: Number of hidden layers used.
    """
    hidden_number = np.random.choice(params["hidden_number"])
    layer_props = params["layer_props"]
    layers = []
    for i in range(hidden_number):
        neurons_number = np.random.choice(layer_props["layer_neurons_number"])
        init_method = np.random.choice(layer_props["init_method"])
        activation = np.random.choice(layer_props["activation"])
        dropout_rate = np.random.choice(layer_props["dropout_rate"])
        batch_normalization = np.random.choice(layer_props["batch_normalization"])

        layers.append(Affine(neurons_number, init_method()))
        if batch_normalization:
            layers.append(BatchNormalization())

        layers.append(activation())

        if i != hidden_number - 1:
            layers.append(Dropout(dropout_rate))

    return layers, hidden_number


class RandomTuner(Tuner):
    """Random search hyperparameter tuning.

    Performs hyperparameter search by randomly sampling from specified ranges,
    useful for exploring large hyperparameter spaces more efficiently than grid search.
    """

    def optimize(
        self,
        x_train,
        x_test,
        t_train,
        t_test,
        output_layer: Affine,
        loss_layer_cls: Type[Loss],
        params,
        n_samples=10,
    ):
        """Perform random search over hyperparameter space.

        Args:
            x_train: Training input data.
            x_test: Test input data.
            t_train: Training target labels.
            t_test: Test target labels.
            output_layer: Output layer specification.
            loss_layer_cls: Loss function class.
            params: Dictionary of hyperparameter ranges to sample from.
            n_samples: Number of random configurations to evaluate (default: 10).

        Returns:
            Dictionary of best found hyperparameters.
        """
        best_accuracy = 0
        self.best_params = {}
        for _ in range(n_samples):

            layers, hidden_number = generate_random_hidden_layers(params)

            layers.append(copy.deepcopy(output_layer))
            network = NeuralNetwork(layers, loss_layer_cls())
            network.init_weights(x_train.shape[1])

            optimizer = get_random_optimizer(params)

            batch_size = np.random.choice(params["batch_size"])
            epochs = np.random.choice(params["epochs"])

            trainer = Trainer(network, optimizer)
            loss, acc = trainer.fit(
                x_train, x_test, t_train, t_test, batch_size, epochs
            )
            accuracy = trainer.evaluate(x_test, t_test)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_trainer = trainer
                self.best_params = {
                    "hidden_number": hidden_number,
                    "optimizer": type(optimizer).__name__,
                    "learning_rate": optimizer.lr,
                    "beta1": 0 if not isinstance(optimizer, Adam) else optimizer.beta1,
                    "beta2": 0 if not isinstance(optimizer, Adam) else optimizer.beta2,
                    "momentum": (
                        0 if not isinstance(optimizer, Momentum) else optimizer.momentum
                    ),
                    "batch_size": batch_size,
                    "epochs": epochs,
                }
        return self.best_params
