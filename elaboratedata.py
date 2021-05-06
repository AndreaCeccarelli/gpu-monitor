import csv
import pandas as pd
import os
import glob
import time
import numpy as np
from datetime import datetime
import statistics
import error
rows_removable=10 #top and bottom rows to be removed, to avoid bias TODO set to 5
                  #check value of LMS in gpu_monitor
                  #rows_removable *LMS * 2= numero di millisecondi di elaborazione che rimuovo dal log (idealmente; in pratica, nvidia-smi logga quando gli pare)

LOG=['gpu.csv', 'supported-clocks.csv', 'compute-apps.csv', 'retired-pages.csv']
PATH_TO_LOG='datalog/' #datalog/gpu.csv , datalog/supported-clocks.csv, datalog/compute-apps.csv, datalog/retired-pages.csv
PATH_TO_DATASET='dataset/'

def addheader(f, L, PATH_TO_LOG, LOG):
    if(L==PATH_TO_LOG+LOG[0]):
        f.columns=['timestamp','index','accounting.buffer_size',
            #inforom.oem','inforom.ecc','inforom.power',
            'fan.speed','pstate','clocks_throttle_reasons.supported','clocks_throttle_reasons.active',
            'clocks_throttle_reasons.gpu_idle','clocks_throttle_reasons.applications_clocks_setting',
            'clocks_throttle_reasons.sw_power_cap','clocks_throttle_reasons.hw_slowdown',
            'clocks_throttle_reasons.hw_thermal_slowdown','clocks_throttle_reasons.hw_power_brake_slowdown',
            'clocks_throttle_reasons.sync_boost','memory.total','memory.used','memory.free','compute_mode',
            'utilization.gpu','utilization.memory','encoder.stats.sessionCount','encoder.stats.averageFps',
            'encoder.stats.averageLatency','ecc.mode.current','ecc.mode.pending',
            'ecc.errors.corrected.volatile.device_memory','ecc.errors.corrected.volatile.register_file',
            'ecc.errors.corrected.volatile.l1_cache','ecc.errors.corrected.volatile.l2_cache',
            'ecc.errors.corrected.volatile.texture_memory','ecc.errors.corrected.volatile.total',
            'ecc.errors.corrected.aggregate.device_memory','ecc.errors.corrected.aggregate.register_file',
            'ecc.errors.corrected.aggregate.l1_cache','ecc.errors.corrected.aggregate.l2_cache',
            'ecc.errors.corrected.aggregate.texture_memory','ecc.errors.corrected.aggregate.total',
            'ecc.errors.uncorrected.volatile.device_memory','ecc.errors.uncorrected.volatile.register_file',
            'ecc.errors.uncorrected.volatile.l1_cache','ecc.errors.uncorrected.volatile.l2_cache',
            'ecc.errors.uncorrected.volatile.texture_memory','ecc.errors.uncorrected.volatile.total',
            'ecc.errors.uncorrected.aggregate.device_memory','ecc.errors.uncorrected.aggregate.register_file',
            'ecc.errors.uncorrected.aggregate.l1_cache','ecc.errors.uncorrected.aggregate.l2_cache',
            'ecc.errors.uncorrected.aggregate.texture_memory','ecc.errors.uncorrected.aggregate.total',
            'retired_pages.single_bit_ecc.count','retired_pages.double_bit.count','retired_pages.pending',
            'temperature.gpu','power.management','power.draw','power.limit','enforced.power.limit',
            'power.default_limit','power.min_limit','power.max_limit','clocks.current.graphics',
            'clocks.current.sm','clocks.current.memory','clocks.current.video','clocks.applications.graphics',
            'clocks.applications.memory','clocks.default_applications.graphics','clocks.default_applications.memory',
            'clocks.max.graphics','clocks.max.sm','clocks.max.memory']
    elif(L==PATH_TO_LOG+LOG[1]):
        f.columns=['timestamp','memory','graphics']
    elif(L==PATH_TO_LOG+LOG[2]):
        f.columns=['timestamp','pid','process_name','used_gpu_memory']
    elif(L==PATH_TO_LOG+LOG[3]):
        f.columns=['timestamp','retired_pages.address','retired_pages.cause']
    return f

def removeRows(f, n):
    print("total size of file "+str(f.name)+" is "+str(f.shape))
    f.drop(f.tail(n).index,inplace=True)
    f.drop(f.head(n).index,inplace=True)
    print("After removing "+str(n)+" lines at top and bottom it is "+str(f.shape))
    return f

def removeMeasurementUnit(f, L, PATH_TO_LOG, LOG):
    if(L==PATH_TO_LOG+LOG[0]):
        for column in f:
            if(f[column].dtype==np.object):
                f[column]=f[column].replace({' MiB':''}, regex=True)
                f[column]=f[column].replace({' MHz':''}, regex=True)
                f[column]=f[column].replace({' W':''}, regex=True)
                f[column]=f[column].replace({'\%':''}, regex=True)
                f[column]=f[column].replace({'MiB':''}, regex=True)
                f[column]=f[column].replace({'MHz':''}, regex=True)
                f[column]=f[column].replace({'W':''}, regex=True)
                f[column]=f[column].replace({'\%':''}, regex=True)
    if(L=='datalog/supported-clocks.csv'):
        f['memory']=f['memory'].replace({' MHz':''}, regex=True)
        f['graphics']=f['graphics'].replace({' MHz':''}, regex=True)
        f['memory']=f['memory'].replace({'MHz':''}, regex=True)
        f['graphics']=f['graphics'].replace({'MHz':''}, regex=True)
    if(L=='datalog/compute-apps.csv'):
        f['used_gpu_memory']=f['used_gpu_memory'].replace({'MiB':''}, regex=True)
        f['used_gpu_memory']=f['used_gpu_memory'].replace({' MiB':''}, regex=True)
    elif(L=='datalog/retired-pages.csv'):
        return f

    return f

#prende i valori in datalog e li mette in dataset, creando inoltre gli UNIQUE
#fa un UNIQUE per ogni iterazione (itemNumber)
def elaborateData(CLASS, ATTACK, IMAGESET, LEARNER, itemNumber):
    #CLASS='normal' #anomaly
    #ATTACK='normal' #normal, or attack type
    #LEARNER='pytorch' #pytorch, tensorflow
    #IMAGESET='da definire' #MNIST, CISAR, etc
    CSV_NAMES=["csv1","csv2","csv3","csv4"]
    for i in range(4):
        LABEL=PATH_TO_LOG+LOG[i]
        print("reading "+LABEL+" for "+ATTACK+ " on "+ IMAGESET+ " for iteration "+str(itemNumber))
        try:
            file= pd.read_csv(LABEL)
        except:
            continue #ho avuto un problema con quel file, quindi vado avanti libero senza di lui (o almeno ci provo)
        file.name=LOG[i]
        file= removeRows(file, rows_removable) #remove biased rows
        file=addheader(file, LABEL,PATH_TO_LOG, LOG) #add header row

        file=removeMeasurementUnit(file, LABEL, PATH_TO_LOG, LOG)
        file['label'] = CLASS # add anomaly or normal
        file['multilabel'] = ATTACK #add details on attack
        FINAL_NAME=PATH_TO_DATASET+ATTACK+'_on_'+IMAGESET+'_with_'+LEARNER+'_'+LOG[i]+'_'+str(itemNumber)
        file.to_csv(FINAL_NAME, index=False)

        CSV_NAMES[i]=FINAL_NAME
    #ora creo un file unico
    for i in range(4):
        if("gpu" in CSV_NAMES[i]):
            df1=pd.read_csv(CSV_NAMES[i]) #gpu.csv
            namefile1=CSV_NAMES[i]
        elif("compute-apps" in CSV_NAMES[i]):#compute-apps.csv
            df2=pd.read_csv(CSV_NAMES[i])
            namefile2=CSV_NAMES[i]
        elif("retired-pages" in CSV_NAMES[i]):#retired-pages.csv
            df3=pd.read_csv(CSV_NAMES[i])
            namefile3=CSV_NAMES[i]
        elif("supported-clocks" in CSV_NAMES[i]):#supported-clocks.csv
            df4=pd.read_csv(CSV_NAMES[i])
            namefile4=CSV_NAMES[i]

    #check retired pages
    number_of_rows = len(df3.index)
    if(number_of_rows> 0):
        error.errorLog("File "+namefile3+" has retired pages diverso da zero, caso unico, investigare")

    ts1=[]
    ts2=[]
    ts4=[]
    for i in df1['timestamp']:
        ts1.append(datetime.strptime(i, '%Y/%m/%d %H:%M:%S.%f'))
    for i in df2['timestamp']:
        ts2.append(datetime.strptime(i, '%Y/%m/%d %H:%M:%S.%f'))
    for i in df4['timestamp']:
        ts4.append(datetime.strptime(i, '%Y/%m/%d %H:%M:%S.%f'))
    #controlla il maggiore
    if(ts1[0]<ts2[0] or len(ts1)>len(ts2)):
        error.errorLog("File "+namefile1+" minore di "+namefile2+": potrebbe essere la causa dei problemi, verificare esistena NaN")

    listts1=[]
    listts2=[]
    listfinal=[]
    for size_ts1 in range(len(ts1)): #gpu.csv
        for size_ts2 in range(len(ts2)):#compute-apps.csv
            if(ts2[size_ts2]>ts1[size_ts1]): #al primo elemento maggiore, considero il precedente
                listfinal.append([ts2[size_ts2-1], ts1[size_ts1]])
                listts1.append(size_ts1)
                listts2.append(size_ts2-1)
                break

    if(len(listts1)!=len(listts2)):
        error.errorLog('ERRORE 1 - PROBLEMA NELLA CREAZIONE DELLE LISTE; lunghezza differente post-elaborazione di '+namefile1+" "+namefile2)
    if(len(listts1)!=len(df1.index)):
        error.errorLog('ERRORE 1 - PROBLEMA NELLA CREAZIONE DELLE LISTE; lunghezza differente post-elaborazione di '+namefile1)

    npts1 = np.array(listts1)
    npts2 = np.array(listts2)
    df2new=df2.iloc[npts2,:]
    df2new.reset_index(inplace=True)
    df2new.drop(df2new.columns[df2new.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    df2new.drop(['label'],axis = 1, inplace = True)
    df2new.drop(['pid'],axis = 1, inplace = True)
    df2new.drop(['process_name'],axis = 1, inplace = True)
    df2new.drop(['multilabel'],axis = 1, inplace = True)
    df2c=df2new.rename(columns={"timestamp": "timestamp_app", "used_gpu_memory": "used_gpu_memory_app"}, errors="raise")
    r = pd.concat([df1, df2c], axis=1)
    df4.drop(df4.columns[df4.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    df4.drop(df4.columns[df4.columns.str.contains('label',case = False)],axis = 1, inplace = True)
    df4.drop(df4.columns[df4.columns.str.contains('multilabel',case = False)],axis = 1, inplace = True)
    d = {'timestamp' 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    df4final = pd.DataFrame(columns=['timestamp_clock', 'mem_clock_min','mem_clock_max','mem_clock_avg','mem_clock_std',
                                                  'graphics_clock_min','graphics_clock_max','graphics_clock_avg','graphics_clock_std'])

    mem_clock_min=0
    mem_clock_max=0
    mem_clock_avg=0
    mem_clock_std=0
    graphics_clock_min=0
    graphics_clock_max=0
    graphics_clock_avg=0
    graphics_clock_std=0
    graph=[]
    mem=[]
    timestamp_clock=ts4[0]
    for i in range(len(ts4)):
        if (timestamp_clock==ts4[i] and i<len(ts4)-1): #se lo stesso timestamp, raccolgo valori
            graph.append(int(df4.iloc[i]['graphics']))
            mem.append(int(df4.iloc[i]['memory']))
        else:
            graphics_clock_min=min(graph)
            graphics_clock_max=max(graph)
            graphics_clock_avg=(sum(graph) / len(graph) )
            graphics_clock_std=statistics.stdev(graph)
            mem_clock_min=min(mem)
            mem_clock_max=max(mem)
            mem_clock_avg=(sum(mem) / len(mem) )
            mem_clock_std=statistics.stdev(mem)
            #e si mette nel data frame panda
            df4final = df4final.append({'timestamp_clock':timestamp_clock,
                                        'mem_clock_min':mem_clock_min,
                                        'mem_clock_max':mem_clock_max,
                                        'mem_clock_avg':mem_clock_avg,
                                        'mem_clock_std':mem_clock_std,
                                        'graphics_clock_min':graphics_clock_min,
                                        'graphics_clock_max':graphics_clock_max,
                                        'graphics_clock_avg':graphics_clock_avg,
                                        'graphics_clock_std':graphics_clock_std}, ignore_index=True)

            timestamp_clock=ts4[i] #ho finito di elaborare timestamp attuale e passo al successivo
            graph=[]
            mem=[]

    a= pd.to_datetime(r['timestamp'])
    b= pd.to_datetime(df4final['timestamp_clock'])
    h1=r.columns.values
    h2=df4final.columns.values
    last=r


    last["timestamp_clock"]=''
    last['mem_clock_min']=''
    last['mem_clock_max']=''
    last['mem_clock_avg']=''
    last['mem_clock_std']=''
    last['graphics_clock_min']=''
    last['graphics_clock_max']=''
    last['graphics_clock_avg']=''
    last['graphics_clock_std']=''
    count=0
    for i in a:
        min_length=1
        for j in b:
            ln=pd.Timedelta(j-i)
            if(ln.seconds < 1):
                min_length=ln
        indx=a[a == i].index[0]
        indx2=b[b == j].index[0]

        last.loc[indx, 'timestamp_clock'] =df4final.loc[indx2, 'timestamp_clock']
        last.loc[indx, 'mem_clock_min'] =df4final.loc[indx2, 'mem_clock_min']
        last.loc[indx, 'mem_clock_max'] =df4final.loc[indx2, 'mem_clock_max']
        last.loc[indx, 'mem_clock_avg'] =df4final.loc[indx2, 'mem_clock_avg']
        last.loc[indx, 'mem_clock_std'] =df4final.loc[indx2, 'mem_clock_std']
        last.loc[indx, 'graphics_clock_min'] =df4final.loc[indx2, 'graphics_clock_min']
        last.loc[indx, 'graphics_clock_max'] =df4final.loc[indx2, 'graphics_clock_max']
        last.loc[indx, 'graphics_clock_avg'] =df4final.loc[indx2, 'graphics_clock_avg']
        last.loc[indx, 'graphics_clock_std'] =df4final.loc[indx2, 'graphics_clock_std']

    last.drop(last.columns[last.columns.str.contains('index',case = False)],axis = 1, inplace = True)
    last.drop(last.columns[last.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    last.drop(last.columns[last.columns.str.contains('timestamp_clock',case = False)],axis = 1, inplace = True)
    last.drop(last.columns[last.columns.str.contains('timestamp_app',case = False)],axis = 1, inplace = True)

    if not os.path.isfile(PATH_TO_DATASET+ATTACK+'_on_'+IMAGESET+'_with_'+LEARNER+'_UNIQUE_'+str(itemNumber)+'.csv'):
        last.to_csv(PATH_TO_DATASET+ATTACK+'_on_'+IMAGESET+'_with_'+LEARNER+'_UNIQUE_'+str(itemNumber)+'.csv', index=False)
    else: # else it exists so append without writing the header
        last.to_csv(PATH_TO_DATASET+ATTACK+'_on_'+IMAGESET+'_with_'+LEARNER+'_UNIQUE_'+str(itemNumber)+'.csv', mode='a', index=False,header=False)

    #last.to_csv(PATH_TO_DATASET+ATTACK+'_on_'+IMAGESET+'_with_'+LEARNER+'_UNIQUE_'+str(itemNumber)+'.csv', index=False)

#fonde tutti gli UNIQUE.csv in un unico file
def mergeAll(DATASETS, itemNumber):
    extension = 'csv'
    os.chdir(PATH_TO_DATASET)
    combined_csv=pd.DataFrame()
    for i in DATASETS:
        for k in range(itemNumber):
            for files in glob.glob('*'+i+'_with'+'*UNIQUE_'+str(k)+'.{}'.format(extension)):
                print(files)
                f=pd.read_csv(files)
                print(f.shape)
                combined_csv=combined_csv.append(f)
                print(combined_csv.shape)
        combined_csv.to_csv(i+"_COMPLETE_DATASET.csv", index=False, encoding='utf-8-sig')
        #reset pandas dataframe
        combined_csv=pd.DataFrame()

#forse non serve
def createFinal(DATASETS, itemNumber):
    extension = 'csv'
    os.chdir(PATH_TO_DATASET)
    combined_csv=pd.DataFrame()
    for i in DATASETS:
        for files in glob.glob(i+'_combined_normal_attacks.csv'):
            print("files")
            f=pd.read_csv(files)
            print(f.shape)
            combined_csv=combined_csv.append(f)
            print(combined_csv.shape)
        combined_csv.to_csv(i+"_CompleteDataset.csv", index=False, encoding='utf-8-sig')
        #reset pandas dataframe
        combined_csv=pd.DataFrame()
        print(combined_csv.shape)
