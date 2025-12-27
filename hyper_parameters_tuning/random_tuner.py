from hyper_parameters_tuning.tuner import Tuner
from layers.affine import Affine
from layers.optimization.batch_normalization import BatchNormalization
from layers.optimization.dropout import Dropout
from network.neural_network import NeuralNetwork
from network.trainer import Trainer
from optimizers.adam import Adam
from optimizers.momentum import Momentum


import numpy as np


import copy


class RandomTuner(Tuner):

    def get_best_params(
        self,
        x_train,
        x_test,
        t_train,
        t_test,
        output_layer,
        loss_layer,
        params,
        n_samples=10,
    ):
        best_accuracy = 0
        self.best_params = {}
        for _ in range(n_samples):

            layers, hidden_number = self.__generate_random_hidden_layers(params)

            layers.append(copy.deepcopy(output_layer))
            network = NeuralNetwork(layers, copy.deepcopy(loss_layer))
            network.init_weights(x_train.shape[1])

            optimizer = self.__get_random_optimizer(params)

            batch_size = np.random.choice(params["batch_size"])
            epochs = np.random.choice(params["epochs"])

            trainer = Trainer(network, optimizer)
            loss, acc = trainer.fit(
                x_train, x_test, t_train, t_test, batch_size, epochs
            )
            accuracy = trainer.evaluate(x_test, t_test)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_params = {
                    "trainer": trainer,
                    "hidden_number": hidden_number,
                    "network_structure": network.structure(),
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

    def __get_random_optimizer(self, params):
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

    def __generate_random_hidden_layers(self, params):
        hidden_number = np.random.choice(params["hidden_number"])
        layer_props = params["layer_props"]
        layers = []
        for _ in range(hidden_number):
            neurons_number = np.random.choice(layer_props["layer_neurons_number"])
            init_method = np.random.choice(layer_props["init_method"])
            activation = np.random.choice(layer_props["activation"])
            dropout_rate = np.random.choice(layer_props["dropout_rate"])
            batch_normalization = np.random.choice(layer_props["batch_normalization"])

            layers.append(Dropout(dropout_rate))
            layers.append(Affine(neurons_number, init_method()))
            layers.append(activation())
            if batch_normalization:
                layers.append(BatchNormalization())
        return layers, hidden_number
