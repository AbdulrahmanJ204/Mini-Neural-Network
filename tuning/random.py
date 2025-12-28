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


# todo : format

def get_random_optimizer(params):
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
                self.best_trainer= trainer
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
