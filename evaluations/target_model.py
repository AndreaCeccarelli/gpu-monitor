#code elaborated starting from B. Lau, The Right Way to Use Deep Learning for Tabular Data | Entity Embedding, https://towardsdatascience.com/the-right-way-to-use-deep-learning-for-tabular-data-entity-embedding-b5c4aaf1423a 

#!/usr/bin/env python
# coding: utf-8

# In[1]:

FILENAME='stl10_COMPLETE_DATASET.csv'
EPOCHS=500 
ATTACKLIST=['FULL', 'AdversarialPatchNumpy', 'SpatialTransformation', 'CarliniL2Method', 'SimBA', 'SquareAttack', 'CarliniLInfMethod', 'allwhite', 'FastGradientMethod', 'BasicIterativeMethod', 'HopSkipJump', 'DeepFool', 'NewtonFool','ProjectedGradientDescentPyTorch','ElasticNet','allblack']
#   'Zoo', 
#remember to remove 'Zoo' from ATTACKLIST in STL
#remember to remove 'AdversarialPatchNumpy' from ATTACKLIST in MNIST

TEST_SIZE_SPLIT= 0.3
TRAIN_VAL_SPLIT=0.2

LOGFILE='/home/XXXXXXXXXX/logfilename.log'
DIR='/home/XXXXX/ORIGINAL/'
PATH=DIR+FILENAME

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Embedding, Dense, Reshape, Concatenate, Dropout, BatchNormalization
from tensorflow.keras import Model
from tensorflow import keras
import tensorflow as tf
import sklearn.metrics as metrics
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import gc

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# In[5]:


label='label' #values 0 and 1 (anomaly/normal)
multilabel=['multilabel']

alwaysdrop=['timestamp', 'accounting.buffer_size',
            #'fan.speed',
            'pstate',
            'clocks_throttle_reasons.supported',
            #'clocks_throttle_reasons.active',
           'clocks_throttle_reasons.gpu_idle',
            'clocks_throttle_reasons.applications_clocks_setting',
            #'clocks_throttle_reasons.sw_power_cap',
            #'clocks_throttle_reasons.hw_slowdown',
            #'clocks_throttle_reasons.hw_thermal_slowdown',
            'clocks_throttle_reasons.hw_power_brake_slowdown',
            'clocks_throttle_reasons.sync_boost','memory.total','memory.used','memory.free',
            'compute_mode','encoder.stats.sessionCount','encoder.stats.averageFps',
            'encoder.stats.averageLatency',
            'ecc.mode.current','ecc.mode.pending','ecc.errors.corrected.volatile.device_memory',
            'ecc.errors.corrected.volatile.register_file','ecc.errors.corrected.volatile.l1_cache',
            'ecc.errors.corrected.volatile.l2_cache','ecc.errors.corrected.volatile.texture_memory',
            'ecc.errors.corrected.volatile.total','ecc.errors.corrected.aggregate.device_memory',
            'ecc.errors.corrected.aggregate.register_file','ecc.errors.corrected.aggregate.l1_cache',
            'ecc.errors.corrected.aggregate.l2_cache','ecc.errors.corrected.aggregate.texture_memory',
            'ecc.errors.corrected.aggregate.total','ecc.errors.uncorrected.volatile.device_memory',
            'ecc.errors.uncorrected.volatile.register_file','ecc.errors.uncorrected.volatile.l1_cache',
            'ecc.errors.uncorrected.volatile.l2_cache','ecc.errors.uncorrected.volatile.texture_memory',
            'ecc.errors.uncorrected.volatile.total','ecc.errors.uncorrected.aggregate.device_memory',
            'ecc.errors.uncorrected.aggregate.register_file','ecc.errors.uncorrected.aggregate.l1_cache',
            'ecc.errors.uncorrected.aggregate.l2_cache','ecc.errors.uncorrected.aggregate.texture_memory',
            'ecc.errors.uncorrected.aggregate.total','retired_pages.single_bit_ecc.count','retired_pages.double_bit.count',
            'retired_pages.pending','power.management','power.limit',
            'enforced.power.limit','power.default_limit','power.min_limit','power.max_limit',
            'clocks.current.memory','clocks.applications.graphics',
            'clocks.applications.memory','clocks.default_applications.graphics','clocks.default_applications.memory',
            'clocks.max.graphics','clocks.max.sm','clocks.max.memory','used_gpu_memory_app',
            'mem_clock_max','graphics_clock_min','graphics_clock_max']

datasetDiscrete=['utilization.gpu',
                 'utilization.memory',
                 'clocks.current.graphics',
                 'clocks.current.sm',
                 'temperature.gpu',
                 'clocks.current.video',
                 'mem_clock_min',
                 'fan.speed',
                 'clocks_throttle_reasons.active',
                 'clocks_throttle_reasons.hw_slowdown',
                 'clocks_throttle_reasons.hw_thermal_slowdown',
                 'clocks_throttle_reasons.sw_power_cap']

datasetContinuous=['power.draw','mem_clock_avg','mem_clock_std','graphics_clock_avg','graphics_clock_std'] #5






cat_vars=datasetDiscrete
cont_vars=datasetContinuous

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc')]


# In[6]:


gpus = tf.config.experimental.list_physical_devices('GPU')

# convert to numerical value for modelling
def categorify(df, cat_vars):
    categories = {}
    for cat in cat_vars:
        df[cat] = df[cat].astype("category").cat.as_ordered()
        categories[cat] = df[cat].cat.categories
    return categories

def apply_test(test,categories):
    for cat, index in categories.items():
        test[cat] = pd.Categorical(test[cat],categories=categories[cat],ordered=True)

# define the neural networks
def combined_network(cat_vars,categories_dict, cont_vars, layers):
    inputs = []
    embeddings = []
    emb_dict ={}
    # create embedding layer for each categorical variables
    for i in range(len(cat_vars)):
        emb_dict[cat_vars[i]] = Input(shape=(1,))
        emb_sz = get_emb_sz(cat_vars[i],categories_dict)
        vocab = len(categories_dict[cat_vars[i]]) +1
        embedding = Embedding(vocab,emb_sz,input_length=1)(emb_dict[cat_vars[i]])
        embedding = Reshape(target_shape=(emb_sz,))(embedding)
        inputs.append(emb_dict[cat_vars[i]])
        embeddings.append(embedding)

    #concat continuous variables
    cont_input = Input(shape=(len(cont_vars),))

    embedding = BatchNormalization()(cont_input)

    inputs.append(cont_input)
    embeddings.append(embedding)
    x = Concatenate()(embeddings)
    # add user-defined fully-connected layers separated with batchnorm and dropout layers
    for i in range(len(layers)):
        if (i == 0):
            x = Dense(layers[i],activation="relu")(x)
        else:
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            x = Dense(layers[i],activation="relu")(x)
    output = Dense(1,activation="sigmoid")(x)
    model = Model(inputs,output)
    print(str(model.summary()))
    return model

#get embeddingsize for each categorical variable
def get_emb_sz(cat_col, categories_dict):
    num_classes=len(categories_dict[cat_col])
    return int(min(600, round(1.6*num_classes**0.56)))


# In[7]:


file_csv=pd.read_csv(PATH)
reduction=file_csv

reduction.drop(alwaysdrop,axis=1,inplace=True) #always drop roba inutile

# In[ ]:

for attack in ATTACKLIST:
    dbTmp=reduction
    if(attack!='FULL'):
        dbTmp1 = dbTmp.loc[dbTmp['multilabel']==attack]
        dbTmp2=dbTmp.loc[dbTmp['multilabel']=='normal']
        dbTmp2=dbTmp2.sample(frac=len(dbTmp1)/len(dbTmp2)).reset_index(drop=True) # fraction to get a number
                                                                                # of random normal sample equivalent to
                                                                                # the number of attacks

        dbTmp=pd.concat([dbTmp1, dbTmp2])


    dbTmp=dbTmp.drop(multilabel,axis=1,inplace=False) #always drop multilabel
    train, test = train_test_split(dbTmp,shuffle=True, test_size=TEST_SIZE_SPLIT)

    train.loc[train[label]=="normal", label]=0
    train.loc[train[label]=="anomaly", label]=1

    test.loc[test[label]=="normal", label]=0
    test.loc[test[label]=="anomaly", label]=1

    x_train = train[cont_vars+cat_vars].copy()
    y_train = train[label].copy()

    x_test = test[cont_vars+cat_vars].copy()
    y_test = test[label].copy()
            # x_train.drop('label',inplace=True) --> non vedo label che starei usando impunemente
            #x_train.drop('multilabel',inplace=True) --> non vedo label che starei usando impunemente
    train.drop([label],axis=1,inplace=True) #nel dubbio, droppiamo il label
    test.drop([label],axis=1,inplace=True) #nel dubbio, droppiamo il label


    TABLESIZE_TRAIN=int(train.shape[0])
    TABLESIZE_TEST=int(test.shape[0])
    scaler = StandardScaler()
    for i in datasetContinuous:
        if(i in x_train):
            x_train[i] = scaler.fit_transform(x_train[i].values.reshape(-1,1))
            x_test[i] = scaler.transform(x_test[i].values.reshape(-1,1))

    # fill missing
    x_train[cat_vars]= x_train[cat_vars].fillna("NaN")
    x_test[cat_vars]= x_test[cat_vars].fillna("NaN")

    # convert to integers
    categories = categorify(x_train,cat_vars)
    apply_test(x_test,categories)
    for cat in cat_vars:
        x_train[cat] = x_train[cat].cat.codes+1
        x_test[cat] = x_test[cat].cat.codes+1

    layer1=int(sys.argv[1])
    layer2=int(sys.argv[2])

    layers = [layer1,layer2]
    model = combined_network(cat_vars,categories, cont_vars, layers)
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt,loss='binary_crossentropy',
        metrics=["accuracy", METRICS[0], METRICS[1], METRICS[2],METRICS[3],METRICS[4]])

            # process x_train input to fit model
    input_list = []
    for i in cat_vars:
        input_list.append(x_train[i].values)

    a=np.empty([TABLESIZE_TRAIN,1])

    k=0
    for i in cont_vars:
        val=x_train[i].values
        val=np.asarray(val)[np.newaxis]
        val=val.T
        if(k==0):
            a=val
            k=1
        else:
            a=np.concatenate((a, val), axis=1)

    input_list.append(a)
    # modify x_test input to fit model
    test_list = []
    for i in cat_vars:
        test_list.append(x_test[i].values)

    b=np.empty([TABLESIZE_TEST,1])
    k=0
    for i in cont_vars:
        val=x_test[i].values
        val=np.asarray(val)[np.newaxis]
        val=val.T
        if(k==0):
            b=val
            k=1
        else:
            b=np.concatenate((b, val), axis=1)

    test_list.append(b)

    #if no improvement for 3 consecutive epochs, stop
    callback=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
    #fit with split and shuffle during fit
    history=model.fit(input_list,y_train,batch_size=512,epochs=EPOCHS,
                              validation_split=TRAIN_VAL_SPLIT, shuffle=True,
                              callbacks=[callback], verbose=1)

    print("EVALUATING ON TRAIN")
    score=model.evaluate(input_list,y_train, batch_size=512, verbose=1)
    print(str(model.metrics_names))
    print(str(score))
    print("EVALUATE ON TEST")
    score=model.evaluate(test_list,y_test, batch_size=512, verbose=1)
    print("test loss, test acc, tp, fp, tn, fn:", score)
    file = open(LOGFILE, 'a')
    file.flush()
    file.write(attack)
    file.write(","+str(score[1])) #Accuracy
    file.write(","+str(score[2])) #tp
    file.write(","+str(score[3])) #fp
    file.write(","+str(score[4])) #tn
    file.write(","+str(score[5])) #fn
    file.write(", LAYER 1 "+ str(layer1))
    file.write(", LAYER 2 " +str(layer2))
    file.write("\n")
    file.flush()
    file.close()

