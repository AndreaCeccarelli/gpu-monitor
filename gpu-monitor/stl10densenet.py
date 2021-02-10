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
from art.utils import load_dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import append
import torchvision.models as models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, HW=96):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = models.densenet121(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    input_size = HW
    return model_ft, input_size
