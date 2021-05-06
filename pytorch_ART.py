import elaboratedata
import utils
import os
import sys
import argparse
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
from utils import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datasetSTL10
from datasetSTL10 import stl10
from sklearn.utils import shuffle
import append
import resnetMnist
import cifar10bis
import mnist
import mnist3
from mnist3 import Net3, StepLR
from mnist4 import Net4
import cifar_resnet
import cifar_mobilenet
import cifar_densenet
from cifar10 import *
import stl10resnet
import stl10densenet
import stl10mobilenet
import torchvision.models as models

#config parameters
SAVEDMODELS='./savedmodels/' #folder must exist
LEARNER='pytorch' #not used
IMAGESET=['cifar10']#'mnist','mnistbis','mnist3','mnist4','cifar10','cifar10bis','cifar10densenet','cifar10resnet','cifar10mobilenet','stl10','stl10densenet','stl10resnet','stl10mobilenet']
HOME='/home/andrea/gpu-monitor' #path must exist and point to this file directory

def ResNet18():
	return cifar10bis.ResNet(cifar10bis.BasicBlock, [2, 2, 2, 2])

#one entry for each algorithm+dataset
#currently: 1 mnist algorithm, 2 cifar, 1 stl10
def main(args):
	SAVEDMODELS=args.savedmodels_path
	LEARNER=args.learner_name
	HOME=args.home
	all=AllAttacks(CIFAR10BIS=args.cifar10bis_repeat,
					MNIST=args.mnist_repeat,
					CIFAR10=args.cifar10_repeat,
					STL10=args.stl10_repeat,
					FULLATTACKS=args.fullattacks_path,
					SYNTETHICATTACKS=args.synteticattacks_path,
					SAVEDATTACKS=args.attacks_library,
					ITERATION_ON_REPETION=args.full_iterations,
					LOG=args.log_path,
					MNISTBIS=args.mnistbis_repeat,
					MNIST3=args.mnist3_repeat,
					MNIST4=args.mnist4_repeat,
					STL10RESNET=args.stl10resnet_repeat,
					STL10DENSENET=args.stl10densenet_repeat,
					STL10MOBILENET=args.stl10mobilenet_repeat,
					CIFAR10RESNET=args.cifar10resnet_repeat,
					#CIFAR10RESNET_DATAUG_NOART=args.cifar10resnet_repeat,#to be checked
					#CIFAR10RESNET_BASE=args.cifar10resnet_repeat,#to be checked
					#CIFAR10RESNET_INPUTNORMALIZED=args.cifar10resnet_repeat,#to be checked
					#CIFAR10RESNET_DATAUG_INART=args.cifar10resnet_repeat,#to be checked
					CIFAR10MOBILENET=args.cifar10mobilenet_repeat,
					CIFAR10DENSENET=args.cifar10densenet_repeat)
	elaboratedata.rows_removable=args.rows_to_remove
	elaboratedata.PATH_TO_LOG=args.path_to_log
	elaboratedata.PATH_TO_DATASET=args.path_to_csv
	append.SLEEP=args.sleep

	for i in IMAGESET:
		os.chdir(HOME)
		print("loading dataset "+i)
		if(i=='mnist'): #mnist train or load
			PATH = SAVEDMODELS+'mnist.pth'
			model = mnist.Net()
			(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(i)
			x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
			x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
			criterion = nn.CrossEntropyLoss()
			optimizer = optim.Adam(model.parameters(), lr=0.01)
			classifier = PyTorchClassifier(
				model=model,
				clip_values=(min_pixel_value, max_pixel_value),
				loss=criterion,
				optimizer=optimizer,
				input_shape=(1, 28,28),
				nb_classes=10,
			)
			if(os.path.exists(PATH)):
				print("Loading saved model for "+i)
				model.load_state_dict(torch.load(PATH))
			else:
				print("Training model for dataset "+i)
				classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)
				torch.save(model.state_dict(), PATH)
		elif(i=='mnist3'): #mnist train or load
			PATH = SAVEDMODELS+'mnist3.pth'
			model = Net3()
			(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(i)
			x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
			x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
			optimizer = optim.Adadelta(model.parameters(), lr=1.0)
			scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
			loss = nn.CrossEntropyLoss()

			classifier = PyTorchClassifier(
				model=model,
				clip_values=(min_pixel_value, max_pixel_value),
				loss=loss,
				optimizer=optimizer,
				input_shape=(1,28,28),
				nb_classes=10,
			)
			if(os.path.exists(PATH)):
				print("Loading saved model for "+i)
				model.load_state_dict(torch.load(PATH))
			else:
				print("Training model for dataset "+i)
				classifier.fit(x_train, y_train, batch_size=128, nb_epochs=14)
				torch.save(model.state_dict(), PATH)
		elif(i=='mnist4'): #mnist train or load
			PATH = SAVEDMODELS+'mnist4.pth'
			model = Net3()
			(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset(i)
			x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
			x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
			optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
			loss = nn.CrossEntropyLoss()
			classifier = PyTorchClassifier(
				model=model,
				clip_values=(min_pixel_value, max_pixel_value),
				loss=loss,
				optimizer=optimizer,
				input_shape=(1,28,28),
				nb_classes=10,
			)
			if(os.path.exists(PATH)):
				print("Loading saved model for "+i)
				model.load_state_dict(torch.load(PATH))
			else:
				print("Training model for dataset "+i)
				classifier.fit(x_train, y_train, batch_size=128, nb_epochs=30)
				torch.save(model.state_dict(), PATH)
		elif(i=='cifar10'): # cifar train or load
			PATH = SAVEDMODELS+'cifar10.pth'
			model = Net1()
			criterion = nn.CrossEntropyLoss()
			optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
			classes = ('plane', 'car', 'bird', 'cat',
				   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
			(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
			x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
			x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
			classifier = PyTorchClassifier(
				model=model,
				clip_values=(min_pixel_value, max_pixel_value),
				loss=criterion,
				optimizer=optimizer,
				input_shape=(3,32,32),
				nb_classes=10,
			)
			if(os.path.exists(PATH)):
				print("Loading saved model for "+i)
				model.load_state_dict(torch.load(PATH))
			else:
				print("Training model for dataset "+i)
				classifier.fit(x_train, y_train, batch_size=1024, nb_epochs=300)
				torch.save(model.state_dict(), PATH)
			#classifier=optimizer
		elif(i=='cifar10bis'): # cifar sperabilmente migliore
			#https://github.com/kuangliu/pytorch-cifar/tree/master/models
			PATH = SAVEDMODELS+'cifar10bis.pth'
			model = ResNet18()
			#model = model.to('cuda')
			criterion = nn.CrossEntropyLoss()
			optimizer = optim.SGD(model.parameters(), lr=0.1,
						  momentum=0.9, weight_decay=5e-4)
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

			(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
			x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
			x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
			classes = ('plane', 'car', 'bird', 'cat', 'deer',
					   'dog', 'frog', 'horse', 'ship', 'truck')
			classifier = PyTorchClassifier(
				model=model,
				clip_values=(min_pixel_value, max_pixel_value),
				loss=criterion,
				optimizer=optimizer,
				input_shape=(3,32,32),
				nb_classes=10,
			)
			if(os.path.exists(PATH)):
				print("Loading saved model for "+i)
				model.load_state_dict(torch.load(PATH))
			else:
				print(PATH)
				print("Training model for dataset "+i)
				classifier.fit(x_train, y_train, batch_size=128, nb_epochs=200)
				torch.save(model.state_dict(), PATH)
		elif(i=='stl10'):#https://github.com/aaron-xichen/pytorch-playground/blob/master/stl10/dataset.py
			PATH = SAVEDMODELS+'stl10.pth'
			a =load_dataset('stl10')
			y_train=np.asarray(a[0][1])
			y_test=np.asarray(a[1][1])
			x_train=np.asarray(a[0][0])
			x_test=np.asarray(a[1][0])
			min=a[2]
			max=a[3]
			x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
			x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
			model = stl10(n_channel=32)
			model = torch.nn.DataParallel(model, device_ids= range(1))
			optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00)
			loss = nn.CrossEntropyLoss()
			classifier = PyTorchClassifier(
				model=model,
				clip_values=(min, max),
				loss=loss,
				optimizer=optimizer,
				input_shape=(3,96,96),
				nb_classes=10,
			)
			if(os.path.exists(PATH)):
				print("Loading saved model for "+i)
				model.load_state_dict(torch.load(PATH))
			else:
				print("Training model for dataset "+i)
				classifier.fit(x_train, y_train, batch_size=256, nb_epochs=50)
				torch.save(model.state_dict(), PATH)
		elif(i=='mnistbis'):
			PATH = SAVEDMODELS+'mnistbis.pth'
			batch_size = 128
			num_epochs = 30 #15 97,5 #20 97,61 #40 97,5 #30 98 #int(num_epochs)
			(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset('mnist')
			x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
			x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

			model = resnetMnist.ResNet()
			optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

			criterion = nn.CrossEntropyLoss()
			classifier = PyTorchClassifier(
				model=model,
				clip_values=(min_pixel_value, max_pixel_value),
				loss=criterion,
				optimizer=optimizer,
				input_shape=(1, 28,28),
				nb_classes=10,
			)
			if(os.path.exists(PATH)):
				print("Loading saved model for "+i)
				model.load_state_dict(torch.load(PATH))
			else:
				print("Training model for dataset "+i)
				classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=num_epochs)
				torch.save(model.state_dict(), PATH)
		elif(i=='stl10resnet'):#https://github.com/aaron-xichen/pytorch-playground/blob/master/stl10/dataset.py
			PATH = SAVEDMODELS+'stl10resnet.pth'
			a =load_dataset('stl10')
			y_train=np.asarray(a[0][1])
			y_test=np.asarray(a[1][1])
			x_train=np.asarray(a[0][0])
			x_test=np.asarray(a[1][0])
			min=a[2]
			max=a[3]
			x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
			x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
			feature_extract = True
			use_pretrained = True
			batch=64
			epoch=50 #0.77875 20 epoch #0.780375 10 epoch #0.760375 5 epoch #0.78675 50 epoch # 0.77925 100 epoch
			model_name = "resnet"
			num_classes = 10
			HW=96 # 32 28
			model, input_size = stl10resnet.initialize_model(model_name, num_classes, feature_extract, use_pretrained, HW)
			device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			model = model.to(device)
			params_to_update = model.parameters()
			print("Params to learn:")
			if feature_extract:
				params_to_update = []
				for name,param in model.named_parameters():
					if param.requires_grad == True:
						params_to_update.append(param)
						print("\t",name)
			else:
				for name,param in model.named_parameters():
					if param.requires_grad == True:
						print("\t",name)
			# Observe that all parameters are being optimized
			optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
			criterion = nn.CrossEntropyLoss()
			classifier = PyTorchClassifier(
				model=model,
				clip_values=(min, max),
				loss=criterion,
				optimizer=optimizer_ft,
				input_shape=(3,HW,HW),
				nb_classes=10,
			)
			#classifier.fit(x_train, y_train, batch_size=batch, nb_epochs=epoch)
			if(os.path.exists(PATH)):
				print("Loading saved model for "+i)
				model.load_state_dict(torch.load(PATH))
			else:
				print("Training model for dataset "+i)
				classifier.fit(x_train, y_train, batch_size=batch, nb_epochs=epoch)
				torch.save(model.state_dict(), PATH)
		elif(i=='stl10densenet'):#https://github.com/aaron-xichen/pytorch-playground/blob/master/stl10/dataset.py
			PATH = SAVEDMODELS+'stl10densenet.pth'
			a =load_dataset('stl10')
			y_train=np.asarray(a[0][1])
			y_test=np.asarray(a[1][1])
			x_train=np.asarray(a[0][0])
			x_test=np.asarray(a[1][0])
			min=a[2]
			max=a[3]
			x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
			x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
			feature_extract = True
			use_pretrained = True
			batch=64
			epoch=40 #  40 epoch 0.827125; 50 epoch 0.821875 ; 10 epoch 0.81575; 100 epoch 0.816375: 30 epoch 0.819125
			model_name = "densenet"
			num_classes = 10
			HW=96 # 32 28
			model, input_size = stl10densenet.initialize_model(model_name, num_classes, feature_extract, use_pretrained, HW)
			device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			model = model.to(device)
			params_to_update = model.parameters()
			print("Params to learn:")
			if feature_extract:
				params_to_update = []
				for name,param in model.named_parameters():
					if param.requires_grad == True:
						params_to_update.append(param)
						print("\t",name)
			else:
				for name,param in model.named_parameters():
					if param.requires_grad == True:
						print("\t",name)
			# Observe that all parameters are being optimized
			optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
			criterion = nn.CrossEntropyLoss()
			classifier = PyTorchClassifier(
				model=model,
				clip_values=(min, max),
				loss=criterion,
				optimizer=optimizer_ft,
				input_shape=(3,HW,HW),
				nb_classes=10,
			)
			#classifier.fit(x_train, y_train, batch_size=batch, nb_epochs=epoch)
			if(os.path.exists(PATH)):
				print("Loading saved model for "+i)
				model.load_state_dict(torch.load(PATH))
			else:
				print("Training model for dataset "+i)
				classifier.fit(x_train, y_train, batch_size=batch, nb_epochs=epoch)
				torch.save(model.state_dict(), PATH)
		elif(i=='stl10mobilenet'):#https://github.com/aaron-xichen/pytorch-playground/blob/master/stl10/dataset.py
			PATH = SAVEDMODELS+'stl10mobilenet.pth'
			a =load_dataset('stl10')
			y_train=np.asarray(a[0][1])
			y_test=np.asarray(a[1][1])
			x_train=np.asarray(a[0][0])
			x_test=np.asarray(a[1][0])
			min=a[2]
			max=a[3]
			x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
			x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
			feature_extract = True
			use_pretrained = True
			batch=64
			epoch=30 #30 epoch fa 0.815125; 10 epoch fa 0.804875; 20 epoch fa 0.8175; 50 epoch fa 0.8165; 100 epoch fa 0.8165
			model_name = "mobilenet"
			num_classes = 10
			HW=96 # 32 28
			model, input_size = stl10mobilenet.initialize_model(model_name, num_classes, feature_extract, use_pretrained, HW)
			device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			model = model.to(device)
			params_to_update = model.parameters()
			print("Params to learn:")
			if feature_extract:
				params_to_update = []
				for name,param in model.named_parameters():
					if param.requires_grad == True:
						params_to_update.append(param)
						print("\t",name)
			else:
				for name,param in model.named_parameters():
					if param.requires_grad == True:
						print("\t",name)
			# Observe that all parameters are being optimized
			optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
			criterion = nn.CrossEntropyLoss()
			classifier = PyTorchClassifier(
				model=model,
				clip_values=(min, max),
				loss=criterion,
				optimizer=optimizer_ft,
				input_shape=(3,HW,HW),
				nb_classes=10,
			)
			#classifier.fit(x_train, y_train, batch_size=batch, nb_epochs=epoch)
			if(os.path.exists(PATH)):
				print("Loading saved model for "+i)
				model.load_state_dict(torch.load(PATH))
			else:
				print("Training model for dataset "+i)
				classifier.fit(x_train, y_train, batch_size=batch, nb_epochs=epoch)
				torch.save(model.state_dict(), PATH)
		elif(i=='cifar10resnet'):#https://github.com/aaron-xichen/pytorch-playground/blob/master/stl10/dataset.py
			PATH = SAVEDMODELS+'cifar10resnet.pth'
			(x_train, y_train), (xtemp, y_test), min_pixel_value, max_pixel_value = load_dataset('cifar10')
			model = cifar_resnet.ResNet18()
			model = model.to('cuda')
			model = torch.nn.DataParallel(model)
			optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
			criterion = nn.CrossEntropyLoss()
			classes = ('plane', 'car', 'bird', 'cat',
				'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
			classifier = PyTorchClassifier(
				model=model,
				clip_values=(min_pixel_value, max_pixel_value),
				loss=criterion,
				optimizer=optimizer,
				input_shape=(3,32,32),
				nb_classes=10,
			)
			transform_test = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
			testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
			testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=2)
			x_test = next(iter(testloader))[0].numpy()
			if(os.path.exists(PATH)):
				model.load_state_dict(torch.load(PATH)['net'])
			else:
				print("If not already trained, go and train it")
				sys.exit(0)
		elif(i=='cifar10mobilenet'):#https://github.com/aaron-xichen/pytorch-playground/blob/master/stl10/dataset.py
			PATH = SAVEDMODELS+'cifar10mobilenet.pth'
			(x_train, y_train), (xtemp, y_test), min_pixel_value, max_pixel_value = load_dataset('cifar10')
			model = cifar_mobilenet.MobileNetV2()
			model = model.to('cuda')
			model = torch.nn.DataParallel(model)
			optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
			criterion = nn.CrossEntropyLoss()
			classes = ('plane', 'car', 'bird', 'cat',
				'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
			classifier = PyTorchClassifier(
				model=model,
				clip_values=(min_pixel_value, max_pixel_value),
				loss=criterion,
				optimizer=optimizer,
				input_shape=(3,32,32),
				nb_classes=10,
			)
			transform_test = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
			testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
			testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=2)
			x_test = next(iter(testloader))[0].numpy()
			if(os.path.exists(PATH)):
				model.load_state_dict(torch.load(PATH)['net'])
			else:
				print("If not already trained, go and train it")
				sys.exit(0)
		elif(i=='cifar10densenet'):#https://github.com/aaron-xichen/pytorch-playground/blob/master/stl10/dataset.py
			PATH = SAVEDMODELS+'cifar10densenet.pth'
			(x_train, y_train), (xtemp, y_test), min_pixel_value, max_pixel_value = load_dataset('cifar10')
			model = cifar_densenet.DenseNet121()
			model = model.to('cuda')
			model = torch.nn.DataParallel(model)
			optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
			criterion = nn.CrossEntropyLoss()
			classes = ('plane', 'car', 'bird', 'cat',
				'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
			classifier = PyTorchClassifier(
				model=model,
				clip_values=(min_pixel_value, max_pixel_value),
				loss=criterion,
				optimizer=optimizer,
				input_shape=(3,32,32),
				nb_classes=10,
			)
			transform_test = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
			testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
			testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=2)
			x_test = next(iter(testloader))[0].numpy()
			if(os.path.exists(PATH)):
				model.load_state_dict(torch.load(PATH)['net'])
			else:
				print("If not already trained, go and train it")
				sys.exit(0)
		model.eval()
		print('Preparations are completed!')
		#si itera su questo insieme di elaborazioni normali e attacchi per assicurarsi che non ci siano bias durante una certa parte dell'esecuzione
		for k in range(all.ITERATION_ON_REPETION):
			print("Monitoring on benign examples")
			all.normal(i, x_test, y_test, classifier, k)
			if (i!='mnist' and i!='mnistbis'and i!='mnist3'and i!='mnist4'): #adv patch does not make sense on mnist
				print("Monitoring on adversarial example AdversarialPatchNumpy")
				all.advPatch(i, x_test, y_test, classifier, k)
			print("Monitoring on adversarial example BasicIterativeMethod")
			all.basic(i, x_test, y_test, classifier, k)
			print("Monitoring on benign examples")
			all.normal(i, x_test, y_test, classifier, k)
			print("Monitoring on adversarial example Carlini L2")
			all.carlini(i, x_test, y_test, classifier, k)
			print("Monitoring on benign examples")
			all.normal(i, x_test, y_test, classifier, k)
			print("Monitoring on adversarial example CarliniLInfMethod")
			all.carliniInf(i, x_test, y_test, classifier, k)
			print("Monitoring on benign examples")
			all.normal(i, x_test, y_test, classifier, k)
			print("Monitoring on adversarial example DeepFool")
			all.deepF(i, x_test, y_test, classifier, k)
			print("Monitoring on benign examples")
			all.normal(i, x_test, y_test, classifier, k)
			print("Monitoring on adversarial example Elastic Net")
			all.elasticN(i, x_test, y_test, classifier, k)
			print("Monitoring on benign examples")
			all.normal(i, x_test, y_test, classifier, k)
			print("Monitoring on adversarial example FAST GRADIENT")
			all.fastgradient(i, x_test, y_test, classifier, k)
			print("Monitoring on benign examples")
			all.normal(i, x_test, y_test, classifier, k)
			print("Monitoring on adversarial example HopSkipJump")
			all.hopskip(i, x_test, y_test, classifier, k)
			print("Monitoring on benign examples")
			all.normal(i, x_test, y_test, classifier, k)
			print("Monitoring on adversarial example NewtonFool")
			all.newtonfool(i, x_test, y_test, classifier, k)
			print("Monitoring on benign examples")
			all.normal(i, x_test, y_test, classifier, k)
			print("Monitoring on adversarial example Projected Gradient Descent")
			all.pgd(i, x_test, y_test, classifier, k)
			print("Monitoring on benign examples")
			all.normal(i, x_test, y_test, classifier, k)
			print("Monitoring on adversarial example SimBA")
			all.simba(i, x_test, y_test, classifier, k)
			print("Monitoring on benign examples")
			all.normal(i, x_test, y_test, classifier, k)
			print("Monitoring on adversarial example Spatial Transformation")
			all.spatial(i, x_test, y_test, classifier, k)
			print("Monitoring on benign examples")
			all.normal(i, x_test, y_test, classifier, k)
			print("Monitoring on adversarial example SquareAttack")
			all.square(i, x_test, y_test, classifier, k)
			print("Monitoring on benign examples")
			all.normal(i, x_test, y_test, classifier, k)
			if (i!='stl10' and i!='stl10densenet' and i!='stl10mobilenet' and i!='stl10resnet'): #zoo does not work on stl10, overflow or too slow
				print("Monitoring on adversarial example ZOO Attack")
				all.zoo(i, x_test, y_test, classifier, k)
			print("Monitoring on benign examples")
			all.normal(i, x_test, y_test, classifier, k)
			print("Monitoring on ALL WHITE")
			all.allwhite(i, x_test, y_test, classifier, k)
			print("Monitoring on benign examples")
			all.normal(i, x_test, y_test, classifier, k)
			print("Monitoring on ALL BLACK")
			all.allblack(i, x_test, y_test, classifier, k)
			os.system("killall nvidia-smi")
		#bisogna esser sicuri di non lasciare questi processi a giro
		os.system("killall nvidia-smi")
		print("creo i dataset fusi")
		elaboratedata.mergeAll(IMAGESET, all.ITERATION_ON_REPETION)
		os.chdir(HOME) #mergeAll sposta la directory
	os.system("killall nvidia-smi")

if __name__ == "__main__":
	parser=argparse.ArgumentParser()
	#default config allows logging approx 600 rows per iteration
	#this way, just setting --full_iterations =50, we reach 30.000 lines for each attack or normal
	#which is more or less enough for our anomaly detectors
	parser.add_argument('--attacks_library', required=False, default='./fullattacks/', help='can be ./fullattacks/ or ./synteticattacks/ resectively if you want to use the full image set or only those where the attacks is successfull') #fullattacks or syntethic attacks
	parser.add_argument('--full_iterations', required=False, type=int, default=50, help='number of iterations on the entire set of attacks; e.g., if 3, it runs through the attack set for 3 times')
	parser.add_argument('--rows_to_remove', required=False, type=int, default=10, help='remove top and bottom rows logged, to remove some noise')
	parser.add_argument('--synteticattacks_path', required=False,  default='./synteticattacks/', help='the path to the synteticattacks, default is ./synteticattacks/, actually if you just donwload the whole set of data provided there is no real reason to change this')
	parser.add_argument('--fullattacks_path', required=False,  default='./fullattacks/', help='the path to the fullattacks, default is ./fullattacks/, actually if you just donwload the whole set of data provided there is no real reason to change this')
	parser.add_argument('--path_to_log', required=False,  default='datalog/', help='temporary data will be logged here')
	parser.add_argument('--path_to_csv', required=False,  default='dataset/', help='your csv will be saved here')
	parser.add_argument('--savedmodels_path', required=False,  default='./savedmodels/', help='path to the trained models')
	parser.add_argument('--learner_name', required=False,  default='pytorch', help='currently works only with pytorch')
	parser.add_argument('--home', required=False,  default='/home/andrea/gpu-monitor', help='IMPORTANT: configure it to the folder where you put all your python files')
	parser.add_argument('--log_path', required=False,  default='./logs/', help='nice logs will be stored here')
	parser.add_argument('--sleep', required=False,  type=float, default=0.5, help='just a small break when changing to a new attack, to further remove noise')

	parser.add_argument('--mnist_repeat', required=False, type=int, default=450,help='number of iterations on the algorithm we label mnist; if too low, you may experience crashes because it completes before logging is done') #1 pass logs 1.5 rows in gpu.csv
	parser.add_argument('--mnistbis_repeat', required=False,  type=int, default=120, help='number of iterations on the algorithm we label mnistbis; if too low, you may experience crashes because it completes before logging is done')#1 pass logs 5 rows in gpu.csv
	parser.add_argument('--mnist3_repeat', required=False, type=int, default=115,help='number of iterations on the algorithm we label mnist3; if too low, you may experience crashes because it completes before logging is done')#1 pass logs 4 rows in gpu.csv
	parser.add_argument('--mnist4_repeat', required=False, type=int, default=115,help='number of iterations on the algorithm we label mnist4; if too low, you may experience crashes because it completes before logging is done') #1 pass logs 5 rows in gpu.csv
	parser.add_argument('--cifar10_repeat', required=False, type=int, default=125, help='number of iterations on the algorithm we label cifar10; if too low, you may experience crashes because it completes before logging is done')#1 pass logs 5 rows in gpu.csv
	parser.add_argument('--cifar10bis_repeat', required=False, type=int, default=9,help='number of iterations on the algorithm we label cifar10bis; if too low, you may experience crashes because it completes before logging is done') #1 pass logs 65 rows in gpu.csv
	parser.add_argument('--cifar10resnet_repeat', required=False, type=int, default=9,help='number of iterations on the algorithm we label cifar10resnet; if too low, you may experience crashes because it completes before logging is done') #1 pass logs 70 rows in gpu.csv
	parser.add_argument('--cifar10densenet_repeat', required=False, type=int, default=2,help='number of iterations on the algorithm we label cifar10densenet; if too low, you may experience crashes because it completes before logging is done') #1 pass logs 70 rows in gpu.csv
	parser.add_argument('--cifar10mobilenet_repeat', required=False, type=int, default=8,help='number of iterations on the algorithm we label cifar10mobilenet; if too low, you may experience crashes because it completes before logging is done') #1 pass logs 80 rows in gpu.csv
	parser.add_argument('--stl10_repeat', required=False, type=int, default=10,help='number of iterations on the algorithm we label stl10; if too low, you may experience crashes because it completes before logging is done') #1 pass logs 55 rows in gpu.csv
	parser.add_argument('--stl10resnet_repeat', required=False, type=int, default=10,help='number of iterations on the algorithm we label stl10resnet; if too low, you may experience crashes because it completes before logging is done') #1 pass logs 50 rows in gpu.csv
	parser.add_argument('--stl10densenet_repeat', required=False, type=int, default=3,help='number of iterations on the algorithm we label stl10densenet; if too low, you may experience crashes because it completes before logging is done') #1 pass logs 120 rows in gpu.csv
	parser.add_argument('--stl10mobilenet_repeat', required=False, type=int, default=11,help='number of iterations on the algorithm we label stl10mobilenet; if too low, you may experience crashes because it completes before logging is done') #1 pass logs 70 rows in gpu.csv

	args=parser.parse_args()
	main(args)
