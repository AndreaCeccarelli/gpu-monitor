import os
import os.path
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from art.utils import load_dataset
from art.attacks.evasion import *
from art.attacks import *
from art.classifiers import PyTorchClassifier
from art.utils import load_mnist
import elaboratedata
import time
from append import *
import elaboratedata
from elaboratedata import elaborateData
from time import sleep
import pandas as pd
import numpy as np
import datasetSTL10
from sklearn.utils import shuffle
import error

class AllAttacks:
    def __init__(self,CIFAR10BIS, MNIST, CIFAR10, STL10, FULLATTACKS,SYNTETHICATTACKS,
					SAVEDATTACKS,
					ITERATION_ON_REPETION,
					LOG,MNISTBIS,MNIST3,MNIST4,
                    STL10RESNET,STL10DENSENET, STL10MOBILENET,
                     CIFAR10RESNET, CIFAR10MOBILENET, CIFAR10DENSENET):
        #ripetizioni su ciascun attacco, consecutive
        self.REPETITION=pd.DataFrame({"cifar10bis":[CIFAR10BIS],#[16], # meno di 200 righe con singola iterazione; 50 valori normal
                "mnist":[MNIST],#[450], #circa 1.16 valore normal per interazione
                "mnist3":[MNIST3],
                "mnist4":[MNIST4],
                "cifar10":[CIFAR10],#[200],
                "stl10":[STL10],#[14]}) #meno di 150 righe con singola iterazione; 35 valori normal
                "stl10resnet":[STL10RESNET],
                "stl10densenet":[STL10DENSENET],
                "stl10mobilenet":[STL10MOBILENET],
                "mnistbis":[MNISTBIS],
                "cifar10resnet":[CIFAR10RESNET],
                "cifar10mobilenet":[CIFAR10MOBILENET],
                "cifar10densenet":[CIFAR10DENSENET]})
        self.ITERATION_ON_REPETION=ITERATION_ON_REPETION #50 #quante volte ripeto l'insieme totale di attacchi
        self.SYNTETHICATTACKS=SYNTETHICATTACKS
        self.FULLATTACKS=FULLATTACKS
        self.SAVEDATTACKS=SAVEDATTACKS #SYNTETHICATTACKS #FULLATTACKS  #directory where attacks are loaded
        self.SYNTETHICATTACKS=SYNTETHICATTACKS
        self.LOG=LOG
        self.LOGACCURACY=LOG+'accuracy.log'
        self.LOGMETRICS=LOG+'detailed_metrics.log'
        self.LOGMETRICSCSV=LOG+'detailed_metrics.csv'

    #load images and shuffle them
    def loadAttackImages(self, SAVEDATTACKS, ATTACK,IMAGESET,LEARNER, y_test):
        if(self.SAVEDATTACKS==self.FULLATTACKS):
            print("loading test ...")
            x_test_adv=np.load(self.SAVEDATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER+".npy")
        elif(self.SAVEDATTACKS==self.SYNTETHICATTACKS): #attacks only onimages that make everything fail
            print("loading synthetic test ... ")
            x_test_adv=np.load(self.SAVEDATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER+"_x.npy")
            y_test=np.load(self.SAVEDATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER+"_y.npy")
        x_test_adv, y_test = shuffle(x_test_adv, y_test)
        return x_test_adv, y_test

    #elaboration on normal input data
    def normal(self, data, x_test, y_test, classifier, itemN):
        CLASS='normal' #anomaly
        ATTACK='normal' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        print("Running tests on normal data...")
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False): #cosi e'  analogo agli altri, fa gli stessi caricamenti da file
            if(self.SAVEDATTACKS==self.FULLATTACKS):
                np.save(self.SAVEDATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER, x_test)
            else: #synthetic attacks
                np.save(self.SYNTETHICATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER+'_x', x_test)
                np.save(self.SYNTETHICATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER+'_y', y_test)
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,
                                            IMAGESET, LEARNER,y_test)
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN, self.REPETITION, self.LOGACCURACY)

    #from here, there are all the attacks
    def spatial(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='SpatialTransformation' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False):
            if(data=='mnist'):#mnist ha bisogno di una configurazione diversa
                attack = SpatialTransformation(classifier=classifier,
                    max_translation=10,num_translations=5,max_rotation=5, num_rotations=5)
            else: #se cifar10, cifar10bis, stl10
                attack = SpatialTransformation(classifier=classifier,
                    max_translation=5,num_translations=1,max_rotation=5, num_rotations=1)
            createAttackImages(CLASS,self.SAVEDATTACKS,self.FULLATTACKS,self.SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test, attack, classifier,self.LOGMETRICS,self.LOGMETRICSCSV)
        print("attack loaded ... generating adversarial test")
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)

    def deepF(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='DeepFool' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False):
            attack = DeepFool(classifier=classifier, batch_size=128)
            createAttackImages(CLASS,self.SAVEDATTACKS,self.FULLATTACKS,self.SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test,attack,classifier,self.LOGMETRICS,self.LOGMETRICSCSV)
        print("attack loaded ... generating adversarial test")
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)

    def elasticN(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='ElasticNet' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False):
            attack = ElasticNet(classifier=classifier, max_iter=2, batch_size=128)
            createAttackImages(CLASS,self.SAVEDATTACKS,self.FULLATTACKS,self.SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test,attack,classifier,self.LOGMETRICS,self.LOGMETRICSCSV)
        print("attack loaded ... generating adversarial test")
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)

    def basic(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='BasicIterativeMethod' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False):
            if(data=='mnist'):
                attack = BasicIterativeMethod(estimator=classifier, eps=0.1, eps_step=0.02, max_iter=200, batch_size=128)
            else:
                attack = BasicIterativeMethod(estimator=classifier, eps=0.01, eps_step=0.01, max_iter=100, batch_size=128)
            createAttackImages(CLASS,self.SAVEDATTACKS,self.FULLATTACKS,self.SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test,attack,classifier,self.LOGMETRICS,self.LOGMETRICSCSV)
        print("attack loaded ... generating adversarial test")
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER,  y_test, itemN,self.REPETITION,self.LOGACCURACY)

    def pgd(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='ProjectedGradientDescentPyTorch' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False):
            if(data=='mnist'): #config for MNIST
                attack = ProjectedGradientDescent(estimator=classifier, eps=0.1, eps_step=0.1, batch_size=128)
            else: #cifar and stl10
                attack = ProjectedGradientDescent(estimator=classifier, eps=0.01,eps_step=0.01, batch_size=128)
            createAttackImages(CLASS,self.SAVEDATTACKS,self.FULLATTACKS,self.SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test,attack,classifier,self.LOGMETRICS,self.LOGMETRICSCSV)
        print("attack loaded ... generating adversarial test")
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)

    def fastgradient(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='FastGradientMethod' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data  #MNIST, CISAR, etc
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False):
            attack= FastGradientMethod(estimator=classifier, eps=0.09, eps_step=0.0001, targeted=False, minimal=True, batch_size=128)
            createAttackImages(CLASS,self.SAVEDATTACKS, self.FULLATTACKS,self.SYNTETHICATTACKS,ATTACK,IMAGESET,LEARNER,x_test,y_test,attack,classifier,self.LOGMETRICS,self.LOGMETRICSCSV)
        print("attack loaded ...")
        x_test_adv, y_test=self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,y_test)
        print("adversarial tests generated; running tests now")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)

    def carlini(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='CarliniL2Method' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data  #MNIST, CISAR, etc
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False):
            attack = CarliniL2Method(classifier=classifier, batch_size=128)
            createAttackImages(CLASS,self.SAVEDATTACKS,self.FULLATTACKS,self.SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test,attack,classifier,self.LOGMETRICS,self.LOGMETRICSCSV)
        print("attack loaded ... generating adversarial test")
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)

    def newtonfool(self, data, x_test,y_test,  classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='NewtonFool' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False):
            attack = NewtonFool(classifier=classifier, batch_size=128)
            createAttackImages(CLASS,self.SAVEDATTACKS,self.FULLATTACKS,self.SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test,attack,classifier,self.LOGMETRICS,self.LOGMETRICSCSV)
        print("attack loaded ... generating adversarial test")
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)

    def hopskip(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='HopSkipJump' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False):
            attack = HopSkipJump(classifier=classifier, max_iter=5, max_eval=500)
            createAttackImages(CLASS,self.SAVEDATTACKS,self.FULLATTACKS,self.SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test,attack,classifier,self.LOGMETRICS,self.LOGMETRICSCSV)
        print("attack loaded ... generating adversarial test")
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)

    def boundary(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='BoundaryAttack' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False):
            attack = BoundaryAttack(classifier=classifier)
            createAttackImages1(CLASS,self.SAVEDATTACKS,self.FULLATTACKS,self.SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test, y_test, attack,classifier,self.LOGMETRICS,self.LOGMETRICSCSV)
        print("attack loaded ... generating adversarial test")
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER, y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)

    def advPatch(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='AdversarialPatchNumpy' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False):
            scale=0.0
            if(data=='stl10'):
                attack = AdversarialPatchNumpy(classifier=classifier,
                                           target= 0, rotation_max= 180.0, scale_min= 0.1, scale_max= 1.0,
                                           learning_rate= 5.0, max_iter=100,  batch_size = 128)
                print("attack loaded ... now loading or generating patches")
                scale=0.3
            else: #cifar10
                attack = AdversarialPatchNumpy(classifier=classifier,
                                           target= 0, rotation_max= 180.0, scale_min= 0.1, scale_max= 1.0,
                                           learning_rate= 5.0, max_iter=500,  batch_size = 128)
                scale=0.4
            print("attack loaded ... now loading or generating patches")
            createPatches(CLASS,self.SAVEDATTACKS,self.FULLATTACKS,
                    self.SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test, y_test, attack, scale,classifier,
                    self.LOGMETRICS,self.LOGMETRICSCSV)
        x_test_adv, y_test=self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER, y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)

    def carliniInf(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='CarliniLInfMethod' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False):
            attack = CarliniLInfMethod(classifier=classifier,
                                confidence= 0.0,
                                targeted= False,
                                learning_rate= 0.01,
                                max_iter= 10,
                                max_halving = 5,
                                max_doubling = 5,
                                eps = 0.3,
                                batch_size= 128)
            createAttackImages1(CLASS,self.SAVEDATTACKS,self.FULLATTACKS,self.SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test, y_test, attack,classifier,self.LOGMETRICS,self.LOGMETRICSCSV)
        print("attack loaded ... generating adversarial test")
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER, y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)

    def simba(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='SimBA' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False):
            attack = SimBA(classifier=classifier)
            createAttackImagesSIMBA(CLASS,self.SAVEDATTACKS,self.FULLATTACKS,self.SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test, y_test, attack,classifier,self.LOGMETRICS,self.LOGMETRICSCSV)
        print("attack loaded ... generating adversarial test")
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER, y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)

    def square(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='SquareAttack' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False):
            attack = SquareAttack(estimator=classifier,
                        max_iter = 100,
                        eps= 0.1,
                        p_init= 0.8,
                        nb_restarts = 10,
                        batch_size = 128)
            createAttackImages1(CLASS,self.SAVEDATTACKS,self.FULLATTACKS,self.SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test, y_test, attack,classifier,self.LOGMETRICS,self.LOGMETRICSCSV)
        print("attack loaded ... generating adversarial test")
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER, y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)

    def zoo(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='Zoo' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        if(attackExists(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test)==False):
            attack=ZooAttack(classifier= classifier,
                    confidence = 0.8,
                    learning_rate = 0.5,
                    max_iter = 1,
                    binary_search_steps = 1,
                    initial_const = 0.6, abort_early = True, use_resize = False,
                     use_importance = True, nb_parallel = 256,
                    batch_size = 1, variable_h = 0.8)
            createAttackImages1(CLASS,self.SAVEDATTACKS,self.FULLATTACKS,self.SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test, y_test, attack,classifier,self.LOGMETRICS,self.LOGMETRICSCSV)
        print("attack loaded ... generating adversarial test")
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)

    #this dataset has to be created manually and manually copied in synthetic
    def allwhite(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='allwhite' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        print("attacking white images")
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)

    #this dataset has to be created manually and manually copied in synthetic
    def allblack(self, data, x_test, y_test, classifier, itemN):
        CLASS='anomaly' #anomaly
        ATTACK='allblack' #normal, or attack type
        LEARNER='pytorch' #pytorch, tensorflow
        IMAGESET=data #MNIST, CISAR, etc
        print("attacking black images")
        x_test_adv, y_test = self.loadAttackImages(self.SAVEDATTACKS, ATTACK,IMAGESET,LEARNER, y_test)
        print("adversarial tests generated")
        executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,self.REPETITION,self.LOGACCURACY)
