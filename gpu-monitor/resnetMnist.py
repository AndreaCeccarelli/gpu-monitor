
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from art.utils import load_dataset
import art.attacks
from art.classifiers import PyTorchClassifier
from art.utils import load_mnist
from art.utils import load_cifar10
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datasetSTL10
from datasetSTL10 import stl10
from sklearn.utils import shuffle

def tensor_map(x_train,y_train,x_val,y_val): return map(torch.tensor,(x_train,y_train,x_val,y_val))

def preprocess(x):
    return x.view(-1, 1, 28, 28)

def conv(in_size, out_size, pad=1):
    return nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=pad)

class ResBlock(nn.Module):

    def __init__(self, in_size:int, hidden_size:int, out_size:int, pad:int):
        super().__init__()
        self.conv1 = conv(in_size, hidden_size, pad)
        self.conv2 = conv(hidden_size, out_size, pad)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x

    def forward(self, x): return x + self.convblock(x) # skip connection

class ResNet(nn.Module):

    def __init__(self, n_classes=10):
        super().__init__()
        self.res1 = ResBlock(1, 8, 16, 15)
        self.res2 = ResBlock(16, 32, 16, 15)
        self.conv = conv(16, n_classes)
        self.batchnorm = nn.BatchNorm2d(n_classes)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = preprocess(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.maxpool(self.batchnorm(self.conv(x)))
        return x.view(x.size(0), -1)
