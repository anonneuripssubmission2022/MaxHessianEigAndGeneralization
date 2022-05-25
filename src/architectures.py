import torch.nn as nn
import torch
from typing import List
import torch.nn.functional as F
import math
from math import sqrt
from vgg import *
from mlp import *

# DATASET INFO
class DatasetInfo:
    def __init__(self, dataset_name, num_input_channels, image_size, num_classes):
        self.dataset_name = dataset_name
        self.num_input_channels = num_input_channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_pixels = self.num_input_channels * self.image_size**2

    def get_dataset_name(self):
        return self.dataset_name

    def get_num_input_channels(self):
        return self.num_input_channels

    def get_image_size(self):
        return self.image_size

    def get_num_pixels(self):
        return self.num_pixels

    def get_num_classes(self):
        return self.num_classes  

CIFAR10 = DatasetInfo('cifar10',3,32,10)
FASHION_MNIST = DatasetInfo('fashion',1,28,10)

dset_info = {
            'cifar10': CIFAR10,
            'cifar10_1k': CIFAR10,
            'fashion_mnist': FASHION_MNIST,
            'fashion_mnist_1k': FASHION_MNIST,
}

# GET DATASET INFO (HELPER FUNCTIONS)
def num_input_channels(dataset_name: str) -> int:
    return dset_info[dataset_name].get_num_input_channels()

def image_size(dataset_name: str) -> int:
    return dset_info[dataset_name].get_image_size()

def num_pixels(dataset_name: str) -> int:
    return dset_info[dataset_name].get_num_pixels()

def num_classes(dataset_name: str) -> int:
    return dset_info[dataset_name].get_num_classes()



# ARCHITECTURES    
def load_architecture(arch_id: int, dataset_name: str) -> nn.Module:
    num_classes = dset_info[dataset_name].get_num_classes()
    arch_ids = {
                    'vgg11_no_dropout': vgg11_no_dropout(num_classes),
                    'vgg11_bn_no_dropout': vgg11_bn_no_dropout(num_classes),
                    'mlp_p=0': mlp(0.0, num_classes),
                    'mlp_p=1e-1': mlp(0.1, num_classes),
                    'mlp_p=2e-1': mlp(0.2, num_classes),
                    'mlp_p=5e-1': mlp(0.5, num_classes),
                    'mlp_p=6e-1': mlp(0.6, num_classes),
                    'mlp_p=7e-1': mlp(0.7, num_classes)

    }
    architecture = arch_ids.get(arch_id)
    if not architecture:
        raise NotImplementedError(
            "unknown architecture ID: {}".format(arch_id))
    return architecture
