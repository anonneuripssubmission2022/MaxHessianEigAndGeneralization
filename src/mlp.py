'''
Modified from: //d2l.ai/chapter_multilayer-perceptrons/dropout.html
'''

import math
import torch.nn as nn
import torch.nn.init as init

class mlp(nn.Module):
    def __init__(self, p, num_classes):
        super(mlp, self).__init__()
        num_inputs, num_hiddens1, num_hiddens2, num_outputs = 784, 256, 256, num_classes
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
        
        self.dropout_layer1 = nn.Dropout(p=p)
        self.dropout_layer2 = nn.Dropout(p=p)
        
        self.flatten = nn.Flatten()

    def forward(self, X):
        H1 = self.relu(self.lin1(self.flatten(X)))
        H1 = self.dropout_layer1(H1)
        H2 = self.relu(self.lin2(H1))
        H2 = self.dropout_layer2(H2)
        out = self.lin3(H2)
        return out