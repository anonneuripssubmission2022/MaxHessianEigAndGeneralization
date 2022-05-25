import torch
import numpy as np
from typing import Tuple
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10, FashionMNIST
from math import sqrt
import os
from preprocess_data import load_cifar, load_fashion_mnist

DATASETS = [
    "cifar10", "cifar10_1k", "fashion_mnist", "fashion_mnist_1k"
]

def load_dataset(dataset_name: str):
    if dataset_name == 'cifar10':
        return load_cifar('standardized')
    elif dataset_name == 'fashion_mnist':
        return load_fashion_mnist('standardized')
    elif dataset_name == 'cifar10_1k':
        return load_abridged_dataset('cifar10', 1000)
    elif dataset_name == 'fashion_mnist_1k':
        return load_abridged_dataset('fashion_mnist', 1000)
    
def load_abridged_dataset(dataset_name: str, num_to_keep: int, offset=0, train_only=False):
    train, test = load_dataset(dataset_name)
    keep_every = len(train) // num_to_keep
    if train_only:
        return abridge_dataset(train, keep_every, offset=offset), test
    return abridge_dataset(train, keep_every, offset=offset), abridge_dataset(test, keep_every, offset=offset)
    
def abridge_dataset(dataset: TensorDataset, keep_every: int, offset=0):
    idx_in = (torch.arange(0, len(dataset)) % keep_every) == offset
    return TensorDataset(*[tensor[idx_in] for tensor in dataset.tensors])