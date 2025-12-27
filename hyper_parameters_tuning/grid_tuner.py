from hyper_parameters_tuning.tuner import Tuner
from layers.affine import Affine
from layers.optimization.batch_normalization import BatchNormalization
from layers.optimization.dropout import Dropout
from network.neural_network import NeuralNetwork
from network.trainer import Trainer
from optimizers.adam import Adam
from optimizers.momentum import Momentum


import copy
from itertools import product


class LayerSpecs:
    def __init__(self, params):
        self.neurons_number = params["layer_neurons_number"]
        self.init_method = params["init_method"]
        self.activation = params["activation"]
        self.dropout_rate = params["dropout_rate"]
        self.batch_normalization = params["batch_normalization"]

    def build(self):
        layers = [
            Dropout(self.dropout_rate),
            Affine(self.neurons_number, self.init_method()),
            self.activation(),
        ]

        if self.batch_normalization:
            layers.append(BatchNormalization())
        return layers


class GridTuner(Tuner):
    def __init__(self):
        self._networks = []
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

    def reset(self):
        self._networks = []
        self._possible_layers = []
        self._possible_optimizers = []
        self._hidden_layers = []

    def __generate_layer(self, params):

        self._possible_layers.append(LayerSpecs(params))

    def __generate_possible_layers(self):
        param_lists = [
            self._params["layer_props"][key] for key in self._layer_params_keys
        ]

        for combo in product(*param_lists):
            layer_params = dict(zip(self._layer_params_keys, combo))
            self.__generate_layer(layer_params)

    def __generate_possible_optimizers(self):
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
        for combo in product(self._possible_layers, repeat=n_layers):
            self._hidden_layers.append([layer.build() for layer in combo])

    def __generate_possible_networks(self, output_layer, loss_layer):
        for n_layers in self._params["hidden_number"]:
            self._hidden_layers = []
            self.__generate_hidden_layers(n_layers)
            for hidden_set in self._hidden_layers:
                layers = [layer for collect in hidden_set for layer in collect]
                layers.append(copy.deepcopy(output_layer))

                self._networks.append(NeuralNetwork(layers, copy.deepcopy(loss_layer)))

    def get_best_params(
        self, params, x_train, x_test, t_train, t_test, output_layer, loss_layer
    ):
        self.reset()
        self._params = params
        self.__generate_possible_layers()
        self.__generate_possible_optimizers()
        self.__generate_possible_networks(output_layer, loss_layer)
        best_accuracy = 0
        self.best_params = {}
        for optimizer_o in self._possible_optimizers:
            for network_o in self._networks:
                for batch_size in self._params["batch_size"]:
                    for epoch in self._params["epochs"]:

                        network = copy.deepcopy(network_o)
                        print(x_train.shape[1])
                        network.init_weights(x_train.shape[1])
                        optimizer = copy.deepcopy(optimizer_o)
                        trainer = Trainer(network, optimizer)
                        loss, acc = trainer.fit(
                            x_train, x_test, t_train, t_test, batch_size, epoch
                        )
                        accuracy = trainer.evaluate(x_test, t_test)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            self.best_params = {
                                "trainer": trainer,
                                "hidden_number": len(network.layers) - 1,
                                "network_structure": network.structure(),
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
