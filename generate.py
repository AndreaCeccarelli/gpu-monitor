import pandas as pd
import os
import glob

DATASETS=['mnistbis']
itemNumber=24
PATH_TO_DATASET='/home/andrea/gpu-monitor/dataset/' #'/home/andrea/gpu-monitor/dataset/'

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
        combined_csv.to_csv(i+"_"+str(itemNumber)+"_Iterations_COMPLETE_DATASET.csv", index=False, encoding='utf-8-sig')
        #reset pandas dataframe
        combined_csv=pd.DataFrame()


mergeAll(DATASETS, itemNumber)
