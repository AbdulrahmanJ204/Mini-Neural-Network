import copy

from Network.neuralNetwork import NeuralNetwork
from Network.trainer import Trainer
from layers.activation.Sigmoid import Sigmoid
from layers.activation.Relu import Relu
from layers.activation.Linear import Linear
from layers.initializers.HeNormal import HeNormal
from layers.initializers.SmallGaussian import SmallGaussian
from layers.initializers.XavierNormal import XavierNormal
from layers.Affine import Affine
from layers.loss.SoftMaxWithCrossEntropy import SoftMaxWithCrossEntropy
from layers.optimization.BatchNormalization import BatchNormalization
from layers.optimization.Dropout import Dropout
from optimizers.AdaGrad import AdaGrad
from optimizers.Adam import Adam
from optimizers.Momentum import Momentum
from optimizers.SGD import SGD

from itertools import product


# in classification problems , i want to define the neurons of last layer


class HyperParameterTuner:
    def __init__(self):
        self.networks = []
        self.possible_layers = []
        self.possible_optimizers = []
        self.hidden_layers = []

        self.layer_params_keys = [
            "layer_neurons_number",
            "init_method",
            "dropout_rate",
            "batch_normalization",
        ]
        self.params = {}
        self.match_activation = {
            HeNormal: Relu,
            XavierNormal: Sigmoid,
            SmallGaussian: Sigmoid,
        }

    def generate_layer(self, params):
        def layer_factory():
            layers = [
                Dropout(params["dropout_rate"]),
                Affine(params["layer_neurons_number"], params["init_method"]()),
                self.match_activation[params["init_method"]](),
            ]
            if params["batch_normalization"]:
                layers.append(BatchNormalization())
            return layers

        self.possible_layers.append(layer_factory)

    def generate_possible_layers(self):
        param_lists = [
            self.params["layer_props"][key] for key in self.layer_params_keys
        ]

        for combo in product(*param_lists):
            layer_params = dict(zip(self.layer_params_keys, combo))
            self.generate_layer(layer_params)

    def generate_possible_optimizers(self):
        self.possible_optimizers = []
        for optimizer in self.params["optimizer"]:
            for learning_rate in self.params["learning_rate"]:
                if optimizer == Adam:
                    for beta1 in self.params["beta1"]:
                        for beta2 in self.params["beta2"]:
                            self.possible_optimizers.append(
                                optimizer(lr=learning_rate, beta1=beta1, beta2=beta2)
                            )

                elif optimizer == Momentum:
                    for momentum in self.params["momentum"]:
                        self.possible_optimizers.append(
                            optimizer(momentum=momentum, lr=learning_rate)
                        )

                else:
                    self.possible_optimizers.append(optimizer(lr=learning_rate))

    def generate_hidden_layers(self, n_layers):
        for combo in product(self.possible_layers, repeat=n_layers):
            self.hidden_layers.append([factory() for factory in combo])

    def generate_possible_networks(self, output_layer, loss_layer):
        self.networks = []
        for n_layers in self.params["hidden_number"]:
            self.hidden_layers = []
            self.generate_hidden_layers(n_layers)
            for hidden_set in self.hidden_layers:
                layers = [layer for collect in hidden_set for layer in collect]
                layers.append(copy.deepcopy(output_layer))
                self.networks.append(NeuralNetwork(layers, copy.deepcopy(loss_layer)))

    def tune(self, params, x_train, x_test, t_train, t_test, output_layer, loss_layer):
        self.params = params
        self.generate_possible_layers()
        self.generate_possible_optimizers()
        self.generate_possible_networks(output_layer, loss_layer)
        best_accuracy = 0
        self.best_params = {}
        for optimizer_o in self.possible_optimizers:
            for network_o in self.networks:
                for batch_size in self.params["batch_size"]:
                    for epoch in self.params["epochs"]:
                        network = copy.deepcopy(network_o)
                        print(x_train.shape[1])
                        network.init_weights(x_train.shape[1])
                        # print(network.structure())
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
