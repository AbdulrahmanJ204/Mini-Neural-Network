# Neural Network Library

## Overview 

### Core Implementation

This project builds upon the concepts and base implementations introduced in Lab 5 and Lab 6, extending them into a fully modular neural network library.
 I have:
- Organized the code into structured files and folders
- Made the Neural Network class **dynamic** so layers can be added flexibly instead of being hardcoded

### Dynamic Network Architecture

The neural network is implemented in a fully dynamic way. Instead of hardcoding the architecture, the network receives:
- A list of layers (Affine, Activation, Normalization, Regularization)
- A loss layer (MSE or Softmax with Cross Entropy)

Each layer initializes its parameters lazily once the input size is known.  
The method `network.init_weights(input_size)` propagates the input size through all layers and initializes their parameters accordingly.

#### Neural Network Methods

- `init_weights(input_size)` - Initializes all layer weights
- `predict(x, train_flg)` - Forward pass through network
- `loss(x, t)` - Computes loss for given inputs and targets
- `accuracy(x, t)` - Calculates prediction accuracy
- `gradient(x, t)` - Backpropagation through all layers, returns the gradients
- `structure()` - Returns network architecture as string table

### Layer System

All layers inherit from the `Layer` base class. Each layer is assigned a unique ID, which allows optimizers to correctly track and update parameters and gradients.

#### Affine Layer (Main Modifications)
The Affine Layer was my primary focus:
- Takes `output_size` and an `initializer` (e.g., Xavier, He) in the constructor
- Has an `init_weights` method that takes `input_size` as a parameter
- Uses the initializer to initialize layer weights
- The `init_weights` method is called by the `NeuralNetwork.init_weights()` method

#### Batch Normalization
- Added `init_weights` method
- Rest of the implementation is the same as lab code

#### Other Layers
- Reused core logic from the lab implementations with small structural adjustments for compatibility
- Added **Linear** and **Tanh** activation layers
- Implemented **Mean Squared Error (MSE)** loss
- Most layers follow the lab implementations without major changes

### Weight Initializers

Implemented three initialization strategies:
- He Normal
- Xavier Normal  
- Gaussian initialization

### Optimizer System

Created an abstract `Optimizer` class:
- Has an `update` method that takes a network and gradients
- Updates the network parameters using the gradients
- Implemented optimization algorithms based on the Lab 6 implementations, adapted to work with the generalized layer and parameter system:
  - SGD
  - Momentum
  - AdaGrad
  - Adam

### Trainer

The `Trainer` class:
- Takes a network and optimizer in the constructor
- Call `trainer.fit(...)` to train the network

#### Trainer Methods
- `fit(x_train, x_test, t_train, t_test, batch_size, epochs)` - Trains the network
- `evaluate(x_test, t_test)` - Evaluates on test set
- `train_step(x_batch, t_batch)` - Single training iteration

### Hyperparameter Tuning
The tuning module automates both architecture and training hyperparameter selection.

I've implemented two tuning methods:

**Random Tuner** (easier and faster):
- Pass parameters to the tuner
- Generates random parameter combinations `n_samples` times
- Returns the best parameters (highest accuracy)

**Grid Tuner** (exhaustive):
- Generates every possible combination of parameters
- Tests all combinations
- Takes a long time to run
- Returns the best parameters (highest accuracy)

---

## Design Decisions

- Layers are initialized lazily to allow flexible architectures.
- Optimizers operate on layer IDs to decouple optimization logic from layer implementations.
- The tuning module separates architecture parameters from training parameters for clarity.

---

## Project Structure
```
├── initializers/              # Weight initialization strategies
│   ├── initializer.py        # Abstract base class
│   ├── he_normal.py          
│   ├── small_gaussian.py     
│   └── xavier_normal.py      
│
├── layers/                    # Each layer has forward and backward methods
│   ├── layer.py              # Base layer class
│   ├── affine.py             # Fully connected layer
│   │
│   ├── activation/           
│   │   ├── linear.py         
│   │   ├── relu.py           
│   │   ├── sigmoid.py        
│   │   └── tanh.py           
│   │
│   ├── loss/                 
│   │   ├── loss.py           # Base loss class
│   │   ├── mean_squared_error.py
│   │   └── softmax_with_cross_entropy.py
│   │
│   ├── normalization/        
│   │   └── batch_normalization.py
│   │
│   └── regularization/       
│       └── dropout.py        
│
├── models/                    
│   └── neural_network.py     # Main network class
│
├── optimizers/               
│   ├── optimizer.py          # Abstract base class
│   ├── sgd.py                
│   ├── momentum.py           
│   ├── adagrad.py            
│   └── adam.py               
│
├── training/                 
│   └── trainer.py            # Trains network using optimizer
│
├── tuning/                   
│   ├── tuner.py              # Abstract base class
│   ├── grid.py               # Grid search - tests all combinations
│   └── random.py             # Random search - samples n configurations
│
├── examples/                 
│   ├── mnist_classification.py
│   ├── grid_tuner.py         
│   └── random_tuner.py       
│
└── utils/                    
    ├── data.py               # MNIST data loading helpers
    └── visualization.py      # Loss and accuracy plotting
```
## Installation

```bash
pip install -r requirements.txt
```
## Usage
### Mnist Classification Example

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

# Load data
x_train, x_test, t_train, t_test = fetch_mnist_data()
x_train_normalized, x_test_normalized = normalize_mnist_data(x_train, x_test)

# Create network
net = NeuralNetwork(
    layers=[
        Affine(128, HeNormal()),
        BatchNormalization(),
        Relu(),
        Dropout(0.3),
        Affine(64, XavierNormal()),
        Sigmoid(),
        Affine(10, HeNormal()),
        # we can add more layers.
    ],
    loss_layer=SoftMaxWithCrossEntropy(),
)
# Train
trainer = Trainer(net, Adam(lr=0.001))  # we can use any optimizer
loss_hist, acc_hist = trainer.fit(
    x_train, x_test, t_train, t_test, batch_size=100, epochs=10
)

# Evaluate
test_acc = trainer.evaluate(x_test, t_test)
print(f"Test Accuracy: {test_acc:.4f}\n")

print(net.structure())

plot_single_train(loss_hist, acc_hist, "MNIST Classification with Neural Network")
```
### Grid Tuner Example

```python
from utils import fetch_mnist_data
from tuning import GridTuner

x_train, x_test, t_train, t_test = fetch_mnist_data()
from layers import Affine, SoftMaxWithCrossEntropy, Sigmoid
from initializers import SmallGaussian, XavierNormal
from optimizers import Adam

tuner = GridTuner()
# used small search space to reduce execution time.
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
    "beta1": [0.9],
    "beta2": [0.999],
    "momentum": [0.9],
    "batch_size": [128],
    "epochs": [3],
}
best_params = tuner.optimize(
    x_train=x_train,
    x_test=x_test,
    t_train=t_train,
    t_test=t_test,
    output_layer=Affine(10, SmallGaussian()),
    loss_layer_cls=SoftMaxWithCrossEntropy,
    params=search_space,
)

trainer = tuner.best_trainer

tuner.print_best_params()

print(trainer.network.structure())
print(trainer.evaluate(x_test, t_test))

```
### Random Tuner Example

```python
import numpy as np
import random
from tuning import RandomTuner
from utils import fetch_mnist_data
from layers import Affine, SoftMaxWithCrossEntropy, Relu, Tanh, Sigmoid
from initializers import SmallGaussian, XavierNormal ,HeNormal
from optimizers import Adam, Momentum ,AdaGrad

np.random.seed(42)
random.seed(42)


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
    "beta1": [0.9, 0.2],
    "beta2": [0.999, 0.9],
    "momentum": [0.9, 0, 7],
    "batch_size": [128, 64, 256],
    "epochs": [2, 3, 4, 1],
}
best_params = tuner.optimize(
    x_train=x_train,
    x_test=x_test,
    t_train=t_train,
    t_test=t_test,
    output_layer=Affine(10, SmallGaussian()),
    loss_layer_cls=SoftMaxWithCrossEntropy,
    params=search_space,
    n_samples = 2
)
trainer = tuner.best_trainer

tuner.print_best_params()

print(trainer.network.structure())
print(trainer.evaluate(x_test, t_test))
```


        
