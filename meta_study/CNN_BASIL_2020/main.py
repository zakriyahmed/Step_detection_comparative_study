import os
import sys
import time
import datetime

now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout  # Save the original stdout
        self.log = open(filename, "w", buffering=1)  # Line-buffered file

    def write(self, message):
        self.terminal.write(message)  # Write to console (terminal)
        self.log.write(message)       # Write to file
        self.log.flush()              # Ensure file gets updated immediately

    def flush(self):
        self.terminal.flush()         # Ensure terminal output is flushed
        self.log.flush()  


sys.stdout = Logger(f"logs_{timestamp}.txt")
print('Staring logs\n')

import torch
import pandas as pd # type: ignore
import numpy as np

from train import Train
from test_models import Test

test_set = {0:list(np.arange(0,6)),
            1:list(np.arange(6,12)),
            2:list(np.arange(12,19)),
            3:list(np.arange(19,25)),
            4:list(np.arange(25,32))}

train_set = {0:list(np.arange(6,32)),
            1:list(np.arange(0,6))+list(np.arange(12,32)),
            2:list(np.arange(0,12))+list(np.arange(19,32)),
            3:list(np.arange(0,19))+list(np.arange(25,32)),
            4:list(np.arange(0,25))}

def train_test(test):
    all = list(np.arange(32))
    for i in test:
        all.remove(i)
    return all,list(test)

def run(train_individuals,test_individuals,epochs):
    #root = "C:\\Users\\shah\\Downloads\\SIMUL-dataset-master\\SIMUL-dataset-master\\data\\with_person"
    root =  "/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/"
    model_name = f"CNN_BASIL_{epochs}_" +  "_".join(map(str, test_individuals)) + ".pt"
    pred_name = f"CNN_BASIL_{epochs}_" +  "_".join(map(str, test_individuals)) + ".csv"
    label_name = f"CNN_BASIL_{epochs}_" +  "_".join(map(str, test_individuals)) + ".csv"

    a = Train(root,'cuda',0.0001,epochs,model_name,individuals=train_individuals,test_individuals=test_individuals)
    a.train()
    a.save_model(model_name)

    b = Test(root,'cuda',model_name,individuals=test_individuals)
    _,_,start,end = b.test()
    #print(start_with_patches.shape)
    #clean_start = utils.remove_small_brusts(start_with_patches,10)
    #clean_end = utils.remove_small_brusts(end_with_patches,10)

    #start = utils.remove_patches(start)
    #end = utils.remove_patches(end)
    df = {'start':start,'end':end}
    df = pd.DataFrame(df)

    df.to_csv(pred_name)

    df_label = {'start':b.ground_labels[:,0],'end':b.ground_labels[:,1],'activity':b.activity,'sensor':b.sensor_labels}

    df_label = pd.DataFrame(df_label)
    df_label.to_csv(label_name)


if __name__=='__main__':
    epochs = 30
    for keys,values in test_set.items():
        print("\n\n",keys,"\n\n")
        train_individuals,test_individuals = train_test(test_set[1])

        run(train_individuals,test_individuals,30)

    for i in range(32):
        print("\n\n",i,"\n\n")

        train_individuals,test_individuals = train_test([i])

        run(train_individuals,test_individuals,30)
        


