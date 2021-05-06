import numpy as np
from art.utils import load_dataset
import os
import os.path
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from time import sleep
from art.utils import load_dataset
import art.attacks
from art.classifiers import PyTorchClassifier
from art.utils import load_mnist
from art.utils import load_cifar10
import torchvision
import torchvision.transforms as transforms
from art.utils import load_dataset
import elaboratedata
import utils
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datasetSTL10
from datasetSTL10 import stl10
from sklearn.utils import shuffle

CLASS='anomaly' #anomaly
LEARNER='pytorch' #pytorch, tensorflow
IMAGESET='mnist' #MNIST, CIFAR, STL10s
dimension=[1,28,28]
ATTACK='allwhite' #normal, or attack type
#1, 28, 28
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(IMAGESET)
x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
img = np.zeros(dimension,dtype=np.float32)
img.fill(1.0) # or img[:] = 255
print(x_test.shape[0])
for i in range(x_test.shape[0]):
    x_test[i]=img
x_test_adv=x_test
np.save(ATTACK+'_'+IMAGESET+'_'+LEARNER, x_test_adv)
np.save(ATTACK+'_'+IMAGESET+'_'+LEARNER+'_x', x_test_adv)
np.save(ATTACK+'_'+IMAGESET+'_'+LEARNER+'_y', y_test)

ATTACK='allblack' #normal, or attack type
#1, 28, 28
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(IMAGESET)
x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
img = np.zeros(dimension,dtype=np.float32)
img.fill(0.0) # or img[:] = 255
for i in range(x_test.shape[0]):
    x_test[i]=img
x_test_adv=x_test
np.save(ATTACK+'_'+IMAGESET+'_'+LEARNER+'_x', x_test_adv)
np.save(ATTACK+'_'+IMAGESET+'_'+LEARNER+'_y', y_test)
