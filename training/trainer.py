import random
from models.neural_network import NeuralNetwork
from optimizers.optimizer import Optimizer
import numpy as np


class Trainer:
    """Trainer for neural network model.

    Handles the training loop including batching, optimization updates, and validation.
    """

    def __init__(self, network: NeuralNetwork, optimizer: Optimizer):
        """Initialize trainer.

        Args:
            network: Neural network model to train.
            optimizer: Optimization algorithm to use for updating weights.
        """
        self.network = network
        self.optimizer = optimizer

    def fit(self, x_train, x_test, t_train, t_test, batch_size=100, epochs=10):
        """Train the network on training data.

        Performs mini-batch gradient descent for the specified number of epochs,
        evaluating accuracy on test set after each epoch.

        Args:
            x_train: Training input data of shape (num_train, input_size).
            x_test: Test input data of shape (num_test, input_size).
            t_train: Training target labels of shape (num_train, num_classes).
            t_test: Test target labels of shape (num_test, num_classes).
            batch_size: Number of samples per batch (default: 100).
            epochs: Number of training epochs (default: 10).

        Returns:
            Tuple of (loss_history, accuracy_history) where each is a list tracking
            loss and accuracy values during training.
        """
        if not self.network.initialized:
            input_size = x_train.shape[1]
            self.network.init_weights(input_size)

        loss_hist = []
        accuracy_hist = []
        loss = 0

        def batch_generator(x, t, batch_size):
            """Generate mini-batches from data.

            Args:
                x: Input data.
                t: Target labels.
                batch_size: Size of each batch.

            Yields:
                Tuples of (x_batch, t_batch).
            """
            n = len(x)
            indices = np.arange(n)
            np.random.shuffle(indices)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]
                yield x[batch_idx], t[batch_idx]

        for i in range(epochs):
            for x_batch, t_batch in batch_generator(x_train, t_train, batch_size):
                loss = self.train_step(x_batch, t_batch)
                loss_hist.append(loss)

            acc = self.network.accuracy(x_test, t_test)
            accuracy_hist.append(acc)

            print(f" Epoch {i + 1}, Loss: {loss:.4f}, Acc: {acc:.4f}")

        return loss_hist, accuracy_hist

    def train_step(self, x_batch, t_batch):
        """Single training step on a batch.

        Computes loss and gradients, then updates network parameters.

        Args:
            x_batch: Batch input data.
            t_batch: Batch target labels.

        Returns:
            Loss value for the batch.
        """
        grads = self.network.gradient(x_batch, t_batch)

        self.optimizer.update(self.network, grads)

        loss = self.network.loss(x_batch, t_batch)
        return loss

    def evaluate(self, x_test, t_test):
        """Evaluate network accuracy on test data.

        Args:
            x_test: Test input data.
            t_test: Test target labels.

        Returns:
            Accuracy value in range [0, 1].
        """
        return self.network.accuracy(x_test, t_test)
