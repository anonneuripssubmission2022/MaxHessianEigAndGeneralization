import torch
import numpy as np
from typing import Tuple
from torch.utils.data import TensorDataset, Dataset
from torchvision.datasets import CIFAR10, FashionMNIST
from math import sqrt
import os

DATASETS_FOLDER = os.environ["DATASETS"]
PREPROCESSINGS = ["raw", "centered", "standardized", "pca", "zca"]

# load and save CIFAR10
def load_cifar(preprocessing: str):
    assert preprocessing in PREPROCESSINGS
    location = os.path.join(DATASETS_FOLDER, "cifar_" + preprocessing)
    return load_tensor_dataset(location)

def save_cifar(preprocessing: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    assert preprocessing in PREPROCESSINGS
    location = os.path.join(DATASETS_FOLDER, "cifar_" + preprocessing)
    shape = (32, 32, 3)
    save_tensor_dataset(location, unflatten(X_train, shape), y_train, unflatten(X_test, shape), y_test)
    
# load and save FashionMNIST
def load_fashion_mnist(preprocessing: str):
    assert preprocessing in PREPROCESSINGS
    location = os.path.join(DATASETS_FOLDER, "fashion_mnist_" + preprocessing)
    return load_tensor_dataset(location)

def save_fashion_mnist(preprocessing: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    assert preprocessing in PREPROCESSINGS
    location = os.path.join(DATASETS_FOLDER, "fashion_mnist_" + preprocessing)
    shape = (28,28,1)
    save_tensor_dataset(location, unflatten(X_train, shape), y_train, unflatten(X_test, shape), y_test)

# Preprocessing helper function
def tensor_dataset(data, input_key, label_key):
    inputs = torch.tensor(data[input_key].transpose((0, 3, 1, 2))).float()
    labels = torch.tensor(data[label_key])
    return TensorDataset(inputs, labels)

def load_tensor_dataset(location: str) -> (Dataset, Dataset):
    data = np.load(os.path.join(DATASETS_FOLDER, location + ".npz"))
    train = tensor_dataset(data, 'X_train', 'y_train')
    test = tensor_dataset(data, 'X_test', 'y_test')
    return train, test


def save_tensor_dataset(location: str, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray):
    np.savez(os.path.join(DATASETS_FOLDER, location), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def center(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(0)
    return X_train - mean, X_test - mean


def standardize(X_train: np.ndarray, X_test: np.ndarray):
    std = X_train.std(0)
    return (X_train / std, X_test / std)


def pca_whiten(X_train: np.ndarray, X_test: np.ndarray):
    n = len(X_train)
    U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
    W = sqrt(n) * np.diag(1.0 / S).dot(Vt)
    return X_train.dot(W.T), X_test.dot(W.T)


def zca_whiten(X_train: np.ndarray, X_test: np.ndarray):
    n = len(X_train)
    U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
    W = sqrt(n) * Vt.T.dot(np.diag(1.0 / S)).dot(Vt)
    return X_train.dot(W.T), X_test.dot(W.T)


def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)


def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)

if __name__ == "__main__":
    # preprocess and save CIFAR10
    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)

    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = cifar10_train.targets, cifar10_test.targets

    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    pca_X_train, pca_X_test = pca_whiten(center_X_train, center_X_test)
    zca_X_train, zca_X_test = zca_whiten(center_X_train, center_X_test)

    save_cifar('raw', X_train, y_train, X_test, y_test)
    save_cifar('centered', center_X_train, y_train, center_X_test, y_test)
    save_cifar('standardized', standardized_X_train, y_train, standardized_X_test, y_test)
    save_cifar('pca', pca_X_train, y_train, pca_X_test, y_test)
    save_cifar('zca', zca_X_train, y_train, zca_X_test, y_test)
    
    # preprocess and save FashionMNIST
    fashion_mnist_train = FashionMNIST(root=DATASETS_FOLDER, download=True, train=True)
    fashion_mnist_test = FashionMNIST(root=DATASETS_FOLDER, download=True, train=False)
    
    fashion_mnist_train.data = fashion_mnist_train.data.unsqueeze(3)
    fashion_mnist_test.data = fashion_mnist_test.data.unsqueeze(3)

    X_train, X_test = flatten(fashion_mnist_train.data / 255.0), flatten(fashion_mnist_test.data / 255.0)
    y_train, y_test = fashion_mnist_train.targets, fashion_mnist_test.targets

    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)

    save_fashion_mnist('raw', X_train, y_train, X_test, y_test)
    save_fashion_mnist('centered', center_X_train, y_train, center_X_test, y_test)
    save_fashion_mnist('standardized', standardized_X_train, y_train, standardized_X_test, y_test)
