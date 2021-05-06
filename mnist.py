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

#mnist
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
		self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
		self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
		self.fc_2 = nn.Linear(in_features=100, out_features=10)

	def forward(self, x):
		x = F.relu(self.conv_1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv_2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4 * 4 * 10)
		x = F.relu(self.fc_1(x))
		x = self.fc_2(x)
		return x
