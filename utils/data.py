import pandas
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np





def fetch_mnist_data():
    """Load and preprocess the MNIST handwritten digits dataset.

    Downloads MNIST dataset from OpenML and splits it into training and test sets.
    Returns raw pixel values without normalization.

    Returns:
        Tuple of (x_train, x_test, t_train, t_test) where:
        - x_train: Training input of shape (50000, 784).
        - x_test: Test input of shape (10000, 784).
        - t_train: Training labels (one-hot encoded) of shape (50000, 10).
        - t_test: Test labels (one-hot encoded) of shape (10000, 10).
    """
    print("Loading MNIST dataset...")

    X, y = fetch_openml("mnist_784", return_X_y=True, as_frame=False)
    x_train, x_test, t_train, t_test = train_test_split(
        X, y, test_size=10000, shuffle=False, train_size=0.6
    )
    
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    
    encoder = OneHotEncoder(sparse_output=False)

    t_train = t_train.astype(int)
    t_test = t_test.astype(int)

    t_train = encoder.fit_transform(t_train.reshape(-1, 1))
    t_test = encoder.transform(t_test.reshape(-1, 1))
    return x_train, x_test, t_train, t_test


def normalize_mnist_data(x_train, x_test):
    """Normalize MNIST pixel values to [0, 1] range.

    Divides pixel values (originally 0-255) by 255 to normalize to [0, 1].

    Args:
        x_train: Training input of shape (num_train, 784).
        x_test: Test input of shape (num_test, 784).

    Returns:
        Tuple of (x_train_norm, x_test_norm) with normalized values.
    """
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_train /= 255.0
    x_test /= 255.0
    return x_train, x_test




def fetch_breast_cancer_data():
    """Load and prepare breast cancer dataset for binary classification"""
   
    
    print("Loading Breast Cancer dataset...")
    
    data = load_breast_cancer()
    X = data.data
    y = data.target.reshape(-1, 1)  # binary labels (0 or 1)
    
    x_train, x_test, t_train, t_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, x_test, t_train, t_test
    