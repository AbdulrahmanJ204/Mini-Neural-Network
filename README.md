# Mini Neural Network Library

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular, educational neural network library built from scratch in Python. This project implements fundamental deep learning concepts including dynamic network architectures, various optimization algorithms, and automated hyperparameter tuning.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
  - [MNIST Classification](#mnist-classification)
  - [Binary Classification](#binary-classification)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Project Structure](#project-structure)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

### 🏗️ Dynamic Architecture
- **Flexible layer composition** - Add layers dynamically instead of hardcoded architectures
- **Lazy initialization** - Weights are initialized automatically based on input dimensions
- **Modular design** - Each component is independent and reusable

### 🧠 Supported Layers
- **Dense Layers**: Fully connected (Affine) layers
- **Activation Functions**: ReLU, Sigmoid, Tanh, Linear
- **Normalization**: Batch Normalization
- **Regularization**: Dropout
- **Loss Functions**: Softmax with Cross-Entropy, Mean Squared Error (MSE), Binary Cross-Entropy (BCE)

### ⚡ Optimization Algorithms
- Stochastic Gradient Descent (SGD)
- Momentum
- AdaGrad
- Adam

### 🎯 Weight Initialization Strategies
- He Normal
- Xavier Normal
- Small Gaussian

### 🔧 Hyperparameter Tuning
- **Grid Search** - Exhaustive search over parameter space
- **Random Search** - Efficient random sampling of configurations

## Installation

### Requirements
- Python 3.8 or higher
- NumPy
- Matplotlib (for visualization)
- Scikit (optional : to run examples)

### Setup

```bash
# Clone the repository
git clone https://github.com/abdulrahmanJ204/Mini-neural-network.git
cd Mini-neural-network

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Here's a simple example to get you started:

```python
import numpy as np
from layers import Affine, Relu
from layers.loss import SoftMaxWithCrossEntropy
from models import NeuralNetwork
from training import Trainer
from optimizers import Adam
from initializers import HeNormal

# Create a simple network
network = NeuralNetwork(
    layers=[
        Affine(128, HeNormal()),
        Relu(),
        Affine(64, HeNormal()),
        Relu(),
        Affine(10, HeNormal()),
    ],
    loss_layer=SoftMaxWithCrossEntropy()
)

# Train the network
trainer = Trainer(network, Adam(lr=0.001))
loss_history, accuracy_history = trainer.fit(
    x_train, x_test, t_train, t_test,
    batch_size=100,
    epochs=10
)

# Evaluate
test_accuracy = trainer.evaluate(x_test, t_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

## Architecture

### Design Philosophy

This library follows key design principles:

1. **Modularity** - Each component (layers, optimizers, initializers) is independent
2. **Extensibility** - Easy to add new layers or optimization algorithms
3. **Lazy Initialization** - Network architecture adapts to input dimensions automatically
4. **Educational** - Code is structured to be readable and educational

### Dynamic Network Architecture

The neural network class is fully dynamic:

```python
# Network receives layers and loss function
net = NeuralNetwork(
    layers=[layer1, layer2, ...],
    loss_layer=loss_function
)

# Weights are initialized when input size is known
net.init_weights(input_size)
```

#### Key Methods

| Method | Description |
|--------|-------------|
| `init_weights(input_size)` | Initialize all layer parameters |
| `predict(x, train_flg)` | Forward propagation |
| `loss(x, t)` | Compute loss for given inputs and targets |
| `accuracy(x, t)` | Calculate prediction accuracy |
| `gradient(x, t)` | Backpropagation through all layers |
| `structure()` | Display network architecture as formatted table |

### Layer System

All layers inherit from the base `Layer` class and implement:
- `forward(x, train_flg)` - Forward pass computation
- `backward(dout)` - Backward pass (gradient computation)
- `init_weights(input_size)` - Parameter initialization (if applicable)

Each layer is assigned a unique ID for proper parameter tracking during optimization.

## Usage Examples

### MNIST Classification

Complete example for digit classification:

```python
from layers import Affine, Relu, Sigmoid, Dropout, BatchNormalization
from initializers import HeNormal, XavierNormal
from layers.loss import SoftMaxWithCrossEntropy
from models import NeuralNetwork
from training import Trainer
from optimizers import Adam
from utils import fetch_mnist_data, normalize_mnist_data, plot_single_train
import numpy as np
import random

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

set_seed()

# Load and normalize data
x_train, x_test, t_train, t_test = fetch_mnist_data()
x_train_normalized, x_test_normalized = normalize_mnist_data(x_train, x_test)

# Build network
net = NeuralNetwork(
    layers=[
        Affine(128, HeNormal()),
        BatchNormalization(),
        Relu(),
        Dropout(0.3),
        Affine(64, XavierNormal()),
        Sigmoid(),
        Affine(10, HeNormal()),
    ],
    loss_layer=SoftMaxWithCrossEntropy(),
)

# Train
trainer = Trainer(net, Adam(lr=0.001))
loss_hist, acc_hist = trainer.fit(
    x_train, x_test, t_train, t_test,
    batch_size=100,
    epochs=10
)

# Evaluate and visualize
test_acc = trainer.evaluate(x_test, t_test)
print(f"Test Accuracy: {test_acc:.4f}\n")
print(net.structure())
plot_single_train(loss_hist, acc_hist, "MNIST Classification")
```

### Binary Classification

Example using Breast Cancer dataset:

```python
from layers import Affine, Relu, BatchNormalization, Dropout
from initializers import HeNormal
from layers.loss.bce import BCE
from models import NeuralNetwork
from training import Trainer
from optimizers import Adam
from utils.data import fetch_breast_cancer_data

# Load data
x_train, x_test, t_train, t_test = fetch_breast_cancer_data()

# Create network for binary classification
net = NeuralNetwork(
    layers=[
        Affine(64, HeNormal()),
        BatchNormalization(),
        Relu(),
        Dropout(0.3),
        Affine(32, HeNormal()),
        Relu(),
        Dropout(0.2),
        Affine(16, HeNormal()),
        Relu(),
        Affine(1, HeNormal()),  # Single output for binary classification
    ],
    loss_layer=BCE(),
)

# Train
trainer = Trainer(net, Adam(lr=0.001))
loss_hist, acc_hist = trainer.fit(
    x_train, x_test, t_train, t_test,
    batch_size=32,
    epochs=50
)

# Evaluate
test_acc = trainer.evaluate(x_test, t_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### Hyperparameter Tuning

#### Random Search (Recommended)

Efficient random sampling of hyperparameter combinations:

```python
from tuning import RandomTuner
from utils import fetch_mnist_data
from layers import Affine, SoftMaxWithCrossEntropy, Relu, Sigmoid, Tanh
from initializers import XavierNormal, HeNormal
from optimizers import Adam, Momentum, AdaGrad

x_train, x_test, t_train, t_test = fetch_mnist_data()

tuner = RandomTuner()
search_space = {
    "hidden_number": [2, 3, 5],
    "layer_props": {
        "layer_neurons_number": [20, 50, 100],
        "init_method": [XavierNormal, HeNormal],
        "dropout_rate": [0.0, 0.1, 0.3],
        "activation": [Sigmoid, Tanh, Relu],
        "batch_normalization": [False, True],
    },
    "optimizer": [Adam, Momentum, AdaGrad],
    "learning_rate": [0.001, 0.01, 0.1],
    "batch_size": [64, 128, 256],
    "epochs": [5, 10, 15],
}

best_params = tuner.optimize(
    x_train=x_train,
    x_test=x_test,
    t_train=t_train,
    t_test=t_test,
    output_layer=Affine(10, XavierNormal()),
    loss_layer_cls=SoftMaxWithCrossEntropy,
    params=search_space,
    n_samples=20  # Number of random configurations to try
)

# Access best model
trainer = tuner.best_trainer
tuner.print_best_params()
print(trainer.network.structure())
```

#### Grid Search (Exhaustive)

Tests all possible parameter combinations:

```python
from tuning import GridTuner

tuner = GridTuner()
# Define smaller search space for faster execution
search_space = {
    "hidden_number": [2, 3],
    "layer_props": {
        "layer_neurons_number": [20, 50],
        "init_method": [XavierNormal],
        "dropout_rate": [0.0, 0.1],
        "activation": [Sigmoid],
        "batch_normalization": [True],
    },
    "optimizer": [Adam],
    "learning_rate": [0.001, 0.01],
    "batch_size": [128],
    "epochs": [3],
}

best_params = tuner.optimize(
    x_train=x_train,
    x_test=x_test,
    t_train=t_train,
    t_test=t_test,
    output_layer=Affine(10, XavierNormal()),
    loss_layer_cls=SoftMaxWithCrossEntropy,
    params=search_space
)
```

## Project Structure

```
mini-neural-network/
│
├── initializers/              # Weight initialization strategies
│   ├── initializer.py        # Abstract base class
│   ├── he_normal.py          # He initialization
│   ├── xavier_normal.py      # Xavier/Glorot initialization
│   └── small_gaussian.py     # Small random initialization
│
├── layers/                    # Neural network layers
│   ├── layer.py              # Base layer class
│   ├── affine.py             # Fully connected layer
│   │
│   ├── activation/           # Activation functions
│   │   ├── linear.py
│   │   ├── relu.py
│   │   ├── sigmoid.py
│   │   └── tanh.py
│   │
│   ├── loss/                 # Loss functions
│   │   ├── loss.py
│   │   ├── mean_squared_error.py
│   │   ├── bce.py
│   │   └── softmax_with_cross_entropy.py
│   │
│   ├── normalization/
│   │   └── batch_normalization.py
│   │
│   └── regularization/
│       └── dropout.py
│
├── models/
│   └── neural_network.py     # Main neural network class
│
├── optimizers/               # Optimization algorithms
│   ├── optimizer.py          # Abstract base class
│   ├── sgd.py               # Stochastic Gradient Descent
│   ├── momentum.py          # SGD with momentum
│   ├── adagrad.py           # Adaptive Gradient Algorithm
│   └── adam.py              # Adaptive Moment Estimation
│
├── training/
│   └── trainer.py            # Training loop and evaluation
│
├── tuning/                   # Hyperparameter optimization
│   ├── tuner.py             # Abstract base class
│   ├── grid.py              # Grid search implementation
│   └── random.py            # Random search implementation
│
├── utils/                    # Utility functions
│   ├── data.py              # Data loading and preprocessing
│   └── visualization.py     # Plotting utilities
│
├── examples/                 # Usage examples
│   ├── mnist_classification.py
│   ├── breast_cancer_binary_classification.py
│   ├── grid_tuner.py
│   └── random_tuner.py
│
├── requirements.txt
└── README.md
```

## Core Concepts

### Lazy Weight Initialization

Instead of requiring input dimensions upfront, layers initialize their weights when first receiving data:

```python
# Layer only knows output dimension initially
layer = Affine(output_size=128, initializer=HeNormal())

# Weights are created when input size becomes known
layer.init_weights(input_size=784)  # Now W shape is (784, 128)
```

### Layer IDs and Parameter Tracking

Each layer receives a unique ID, enabling optimizers to correctly track and update parameters across the network

### Training Loop

The `Trainer` class encapsulates the training process:

1. **Mini-batch creation** - Randomly sample training data
2. **Forward pass** - Compute predictions and loss
3. **Backward pass** - Calculate gradients via backpropagation
4. **Parameter update** - Apply optimizer to update weights
5. **Evaluation** - Test on validation set

## API Reference

### Models

#### `NeuralNetwork`

```python
NeuralNetwork(layers, loss_layer)
```

**Parameters:**
- `layers` (list): List of layer objects
- `loss_layer`: Loss function layer

**Methods:**
- `init_weights(input_size)`: Initialize network parameters
- `predict(x, train_flg=False)`: Forward propagation
- `loss(x, t)`: Compute loss
- `accuracy(x, t)`: Calculate accuracy
- `gradient(x, t)`: Compute gradients via backpropagation
- `structure()`: Return formatted network architecture

### Training

#### `Trainer`

```python
Trainer(network, optimizer)
```

**Methods:**
- `fit(x_train, x_test, t_train, t_test, batch_size, epochs)`: Train the network
- `evaluate(x_test, t_test)`: Evaluate on test set
- `train_step(x_batch, t_batch)`: Single training iteration

### Optimizers

All optimizers inherit from `Optimizer` base class:

```python
# Available optimizers
SGD(lr=0.01)
Momentum(lr=0.01, momentum=0.9)
AdaGrad(lr=0.01)
Adam(lr=0.001, beta1=0.9, beta2=0.999)
```

### Layers

#### Affine Layer
```python
Affine(output_size, initializer)
```

#### Activation Layers
```python
Relu()
Sigmoid()
Tanh()
Linear()
```

#### Normalization
```python
BatchNormalization(momentum=0.9)
```

#### Regularization
```python
Dropout(dropout_ratio=0.5)
```

### Loss Functions

```python
SoftMaxWithCrossEntropy()  # Multi-class classification
BCE()                       # Binary classification
MeanSquaredError()         # Regression
```

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions and classes
- Include unit tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contact

**Abdulrahman J204**
- GitHub: [@AbdulrahmanJ204](https://github.com/abdulrahmanJ204)
- Project Link: [https://github.com/abdulrahmanJ204/Mini-neural-network](https://github.com/abdulrahmanJ204/Mini-neural-network)

---

<div align="center">
  <sub>Built with ❤️ for learning and education</sub>
</div>
