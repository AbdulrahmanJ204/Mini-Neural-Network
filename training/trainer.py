from models.neural_network import NeuralNetwork
from optimizers.optimizer import Optimizer
import numpy as np


class Trainer:
    def __init__(self, network: NeuralNetwork, optimizer: Optimizer):
        self.network = network
        self.optimizer = optimizer

    def fit(self, x_train, x_test, t_train, t_test, batch_size=100, epochs=10):

        if not self.network.initialized:
            input_size = x_train.shape[1]
            self.network.init_weights(input_size)

        training_size = len(x_train)
        iter_per_epoch = training_size // batch_size

        loss_hist = []
        accuracy_hist = []
        loss = 0
        for i in range(epochs):
            for _ in range(iter_per_epoch):
                batch_mask = np.random.choice(training_size, batch_size)
                x_batch = x_train[batch_mask]
                t_batch = t_train[batch_mask]

                loss = self.train_step(x_batch, t_batch)
                loss_hist.append(loss)

            acc = self.network.accuracy(x_test, t_test)
            accuracy_hist.append(acc)
            print(f"Epoch {i + 1}, Loss: {loss:.4f}, Acc: {acc:.4f}")

        return loss_hist, accuracy_hist

    def train_step(self, x_batch, t_batch):
        grads = self.network.gradient(x_batch, t_batch)

        self.optimizer.update(self.network, grads)

        loss = self.network.loss(x_batch, t_batch)
        return loss

    def evaluate(self, x_test, t_test):
        return self.network.accuracy(x_test, t_test)
