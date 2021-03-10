import csv
import pandas as pd
import os
import glob
import time
import numpy as np
from datetime import datetime
import statistics
import error
from elaboratedata import *
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from math import ceil

SLEEP=10

#check if attack file exists
def attackExists(SAVEDATTACKS, ATTACK,IMAGESET,LEARNER,x_test,y_test):
    return os.path.exists(SAVEDATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER+".npy")

#compute detailed metrics, it is executed only when generating synthetic attack subset
#gives some info on the synthetic attack set
def computeMissingMetrics(CLASS, ATTACK, IMAGESET, LEARNER, predictions, y_test, x_test, classifier,LOGMETRICS,LOGMETRICSCSV):
    pred_normal=executeTest(IMAGESET, x_test, classifier, repeat=1) #calcolo i normal
    f=open(LOGMETRICS, 'a+')
    if(os.path.isfile(LOGMETRICSCSV)==False): #se il file non esiste, aggiungo header file
            f1=open(LOGMETRICSCSV, 'a+')
            f1.write("IMAGESET, ALGORITHM, ATTACK, Accuracy, Accuracy normal, MCC, Dimension test set, adversarial == normal, adversarial == normal; normal ==true, adversarial == normal; normal !=true,"+
                "adversarial != normal; normal ==true, adversarial != normal; normal !=true, adversarial == true; normal !=true, adversarial != normal; adversaral !=true\n")
            f1.close()
    f1=open(LOGMETRICSCSV, 'a+')
    pred_normal_max=np.argmax(pred_normal, axis=1) #argmax sui normal
    pred_adv_max= np.argmax(predictions, axis=1) # argmax sugli attacchi
    y_max= np.argmax(y_test, axis=1) #argmax sulla ground truth
    accuracy_adv=np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) #solita accuracy su attacchi
    accuracy_normal=np.sum(np.argmax(pred_normal, axis=1) == np.argmax(y_test, axis=1)) / len(y_test) #solita accuracy su attacchi
    mcc_adv= matthews_corrcoef(y_max, pred_adv_max) #mcc sugli attacchi
    #adversarial == normal (predictions)
    adv1=np.sum(pred_normal_max == pred_adv_max)
    #adversarial == normal; normal ==true
    normTrue=np.sum(pred_normal_max == y_max)
    adv2=((pred_adv_max==pred_normal_max) & (pred_normal_max==y_max))
    #adversarial == normal; normal !=true
    adv3=((pred_adv_max==pred_normal_max) & (pred_normal_max!=y_max))
    #adversarial != normal; normal ==true
    adv4=((pred_adv_max!=pred_normal_max) & (pred_normal_max==y_max))
    #adversarial != normal; normal !true
    adv5=((pred_adv_max!=pred_normal_max) & (pred_normal_max!=y_max))
    #adversarial == true; normal !=true
    adv6=((pred_adv_max==y_max) & (pred_normal_max!=y_max))
    #adversarial != normal; adversaral !=true
    adv7=((pred_adv_max!=pred_normal_max) & (pred_adv_max!=y_max))
    #stampo un po' di roba, creo un csv
    f.write(IMAGESET+ " under "+ATTACK+"\n")
#    f.write("Confusion matrix")
#    f.write("tp="+str(tp)+" fp="+str(fp)+"fn="+str(fn)+" tn="+str(tn)) #stampo confusion matrix sugli attacchi
    f.write("Accuracy= "+str(accuracy_adv)+"\n")
    f.write("Accuracy Normal= "+str(accuracy_normal)+"\n")
    f.write("MCC (ma non ha senso calcolarlo)= "+str(mcc_adv)+"\n")
    f.write("Dimensione test set = "+ str(len(y_test))+"\n")
    f.write("Prediction adversarial == normal: "+ str(np.sum(adv1))+"\n")
    #adversarial == normal; normal ==true
    f.write("Prediction adversarial == normal; normal ==true: "+ str(np.sum(adv2))+"\n")
    #adversarial == normal; normal !=true
    f.write("Prediction adversarial == normal; normal !=true: "+ str(np.sum(adv3))+"\n")
    #adversarial != normal; normal ==true
    f.write("Prediction adversarial != normal; normal ==true: "+ str(np.sum(adv4))+"\n")
    #adversarial != normal; normal !=true
    f.write("Prediction adversarial != normal; normal !=true: "+ str(np.sum(adv5))+"\n")
    #adversarial == true; normal !=true
    f.write("Prediction adversarial == true; normal !=true: "+ str(np.sum(adv6))+"\n")
    #adversarial != normal; adversaral !=true
    f.write("adversarial != normal; adversaral !=true: "+ str(np.sum(adv7))+"\n")
    f.close()
    f1.write(IMAGESET+ " , "+IMAGESET +","+ATTACK+", "+str(accuracy_adv)+","+str(accuracy_normal)+","+str(mcc_adv)+","+str(len(y_test))+","+str(np.sum(adv1))+","+str(np.sum(adv2))+","+str(np.sum(adv3))+","+
        str(np.sum(adv4))+","+str(np.sum(adv5))+","+str(np.sum(adv6))+","+str(np.sum(adv7))+"\n")
    f1.close()
    return pred_normal

#x_test è clean, no attack IMAGESET
#x_test_adv è avversario
def saveAccuracy(CLASS, ATTACK,  IMAGESET, LEARNER, predictions, y_test, LOGACCURACY):
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    pred_max=np.argmax(predictions, axis=1)
    test_max=np.argmax(y_test, axis=1)
    f=open(LOGACCURACY, 'a+')
    accuracy100=accuracy *100
    f.write("Accuracy on test set of "+IMAGESET+ " under "+CLASS+" data ("+ATTACK+" data point) using "+LEARNER+": {}%".format(accuracy * 100))
    f.write("\n")
    print("Accuracy on test set of "+IMAGESET+ " under "+CLASS+" data ("+ATTACK+" data point) using "+LEARNER+": {}%".format(accuracy * 100))
    f.flush()
    f.close()

#create synthetic attack set, only the "dangerous images"
#dimension is the same of the original datasets
def createAttackSetSynthetic(CLASS, ATTACK,  IMAGESET, LEARNER, predictions, y_test, x_test,x_test_adv, classifier, LOGMETRICS,LOGMETRICSCSV,SYNTETHICATTACKS):
    pred_normal=computeMissingMetrics(CLASS, ATTACK,  IMAGESET, LEARNER, predictions, y_test, x_test, classifier,LOGMETRICS,LOGMETRICSCSV)
    pred_normal_max=np.argmax(pred_normal, axis=1) #argmax sui normal
    pred_adv_max= np.argmax(predictions, axis=1) # argmax sugli attacchi
    y_max= np.argmax(y_test, axis=1) #argmax sulla ground truth
    #adversarial != normal; adversaral !=true
    adv7=((pred_adv_max!=pred_normal_max) & (pred_adv_max!=y_max))
    x_fail=[]
    y_fail=[]
    for i in range(len(adv7)):
        if(adv7[i]==True):
            x_fail.append(x_test_adv[i])
            y_fail.append(y_test[i])

    x_test_adv1=x_fail
    y_test_adv1=y_fail
    size_max=x_test.shape[0]
    size_attacks=len(x_test_adv1) #.shape[0]
    repeat=ceil(size_max/size_attacks)
    x_test__tmp=np.empty(x_test.shape) #target dimension as the normal array
    y_test_tmp=np.empty(y_test.shape) #target dimension as the normal array
    x_test_tmp=np.tile(x_test_adv1, (repeat, 1, 1, 1))
    y_test_tmp=np.tile(y_test_adv1, (repeat, 1 ))
    x_test_adv=x_test_tmp[0:size_max,:]
    y_test=y_test_tmp[0:size_max,:]
    print(x_test_adv.shape)
    print(y_test.shape)
    np.save(SYNTETHICATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER+'_x', x_test_adv)
    np.save(SYNTETHICATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER+'_y', y_test)

def createAttackImagesSIMBA(CLASS,SAVEDATTACKS,FULLATTACKS,SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test, y_test, attack,classifier,LOGMETRICS,LOGMETRICSCSV):
    if(SAVEDATTACKS==FULLATTACKS):
        print("generating test ...")
        print("generating test ...")
        x_test_adv = np.empty(x_test.shape)
        print(x_test.shape)
        x_test_adv=np.array(x_test_adv)
        for i in range(y_test.shape[0]):
            print("now elaborating image "+str(i))
            z=attack.generate(x=[x_test[i]], y=[y_test[i]])
            z1=np.array(z)
            x_test_adv[i]=z1[0] #save image set
            print("x_test_adv size is "+str(x_test_adv.shape))
            print("z[i] for cell "+str(i)+" is "+str(z1[0].shape))
        x_test_adv1=x_test_adv.astype(np.float32)
        np.save(SAVEDATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER, x_test_adv1)
    elif(SAVEDATTACKS==SYNTETHICATTACKS): #attacks only onimages that make everything fail
        print("generating test ...")
        x_test_adv =np.load(FULLATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER+".npy") #save image set
        predictions=classifier.predict(x_test_adv)
        createAttackSetSynthetic(CLASS,ATTACK, IMAGESET, LEARNER, predictions, y_test, x_test, x_test_adv, classifier,LOGMETRICS,LOGMETRICSCSV, SYNTETHICATTACKS)

#create attack dataset, either full or synthetic
def createAttackImages(CLASS,SAVEDATTACKS,FULLATTACKS,SYNTETHICATTACKS,  ATTACK,IMAGESET,LEARNER,x_test, y_test, attack, classifier,LOGMETRICS,LOGMETRICSCSV):
    if(SAVEDATTACKS==FULLATTACKS):
        print("generating test ...")
        x_test_adv = attack.generate(x=x_test) #save image set
        np.save(SAVEDATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER, x_test_adv)
    elif(SAVEDATTACKS==SYNTETHICATTACKS): #attacks only onimages that make everything fail
        print("generating test ...")
        x_test_adv =np.load(FULLATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER+".npy") #save image set
        predictions=classifier.predict(x_test_adv)
        createAttackSetSynthetic(CLASS,ATTACK, IMAGESET, LEARNER, predictions, y_test, x_test, x_test_adv, classifier,LOGMETRICS,LOGMETRICSCSV, SYNTETHICATTACKS)

#create attack dataset, either full or synthetic (but for some attacks which have a different "generate" function)
def createAttackImages1(CLASS,SAVEDATTACKS,FULLATTACKS,SYNTETHICATTACKS, ATTACK,IMAGESET,LEARNER,x_test, y_test, attack, classifier,LOGMETRICS,LOGMETRICSCSV):
    if(SAVEDATTACKS==FULLATTACKS):
        print("generating test ...")
        x_test_adv = attack.generate(x=x_test, y=y_test) #save image set
        np.save(SAVEDATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER, x_test_adv)
    elif(SAVEDATTACKS==SYNTETHICATTACKS): #attacks only onimages that make everything fail
        print("generating test ...")
        x_test_adv =np.load(FULLATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER+".npy") #save image set
        predictions=classifier.predict(x_test_adv)
        createAttackSetSynthetic(CLASS,ATTACK, IMAGESET, LEARNER, predictions, y_test, x_test, x_test_adv, classifier,LOGMETRICS,LOGMETRICSCSV, SYNTETHICATTACKS)

#create the patches to be applied to images, and create the attack set with patched images
def createPatches(CLASS,SAVEDATTACKS,FULLATTACKS,SYNTETHICATTACKS, ATTACK,
                        IMAGESET,LEARNER,x_test, y_test, attack, scale, classifier,LOGMETRICS,LOGMETRICSCSV):
    if(SAVEDATTACKS==FULLATTACKS):
        print("generating test ...")
        print("generating a total of "+str(y_test.shape[0])+" patches")
        patch, patch_mask = attack.generate(x=x_test, y=y_test)
        np.save(SAVEDATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER+'patch', patch)
        np.save(SAVEDATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER+'patch_mask', patch_mask)
        x_test_adv = np.empty(x_test.shape)
        for i in range(y_test.shape[0]):
            print("applying patch to image"+str(i))
            patched_image = attack.apply_patch(np.array([x_test[i]]), patch_external=patch, mask=patch_mask, scale=scale)
            z1=np.array(patched_image)
            x_test_adv[i]=z1[0] #save image set
            print("x_test_adv size is "+str(x_test_adv.shape))
            print("z[i] for cell "+str(i)+" is "+str(z1[0].shape))
        x_test_adv1=x_test_adv.astype(np.float32)
        np.save(SAVEDATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER, x_test_adv1)
    elif(SAVEDATTACKS==SYNTETHICATTACKS): #attacks only onimages that make everything fail
        x_test_adv =np.load(FULLATTACKS+ATTACK+'_'+IMAGESET+'_'+LEARNER+'.npy') #save image set
        predictions=classifier.predict(x_test_adv)
        createAttackSetSynthetic(CLASS,ATTACK, IMAGESET, LEARNER, predictions, y_test, x_test, x_test_adv, classifier,LOGMETRICS,LOGMETRICSCSV, SYNTETHICATTACKS)

def executeTest(IMAGESET, x_test_set, classifier,repeat):
    os.system('rm ./datalog/*')
    os.system("./gpu_monitor.sh &")
    for i in range(repeat):
        predictions = classifier.predict(x_test_set)
    os.system("killall nvidia-smi")
    time.sleep(SLEEP)#a sleep just to allow killall to terminate
    return predictions

#prediction, accuracy computation, logging
def executeAll(IMAGESET, x_test_adv, classifier,CLASS, ATTACK, LEARNER, y_test, itemN,REPETITION,LOGACCURACY):
    repeat=REPETITION[IMAGESET][0]
    predictions=executeTest(IMAGESET, x_test_adv, classifier,repeat)
    saveAccuracy(CLASS, ATTACK,  IMAGESET, LEARNER, predictions, y_test, LOGACCURACY)
    elaborateData(CLASS, ATTACK, IMAGESET, LEARNER, itemN)
    os.system('rm ./datalog/*')
