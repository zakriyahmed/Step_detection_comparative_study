import os
import sys
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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


sys.stdout = Logger("logs.txt")
print('Staring logs\n')

import pandas as pd # type: ignore 
import torch
import numpy as np

def scores(predictions,labels,threshold):
    
    #print(predicted_middle.shape,predicted_middle[0:10])
    ground_truth_indices = torch.nonzero(labels).squeeze()
    #print(ground_truth_indices.shape,ground_truth_indices[0:10])

    pred = torch.nonzero(predictions)
    total_predicted = len(pred)
    #print(pred.shape)
    if len(pred)==0:
        return 0,0,0,np.zeros(threshold).tolist,0
    
    dictionary = {key.item(): 0 for key in ground_truth_indices}
    
    for key,value in dictionary.items():
        minimum_distance = (abs(pred-key)).min()
        dictionary[key]=minimum_distance.item()

    scores = torch.tensor(list(dictionary.values()))
    scores[scores>=threshold]=-1
    va,cou = torch.unique(scores,return_counts=True)
    mae = ((va[1:]*cou[1:]).sum()/cou[1:].sum()).item()  # Expected value of each count/distance
    acc = (cou[1:].sum()/len(dictionary)).item()
    histogram = cou[1:].numpy()
    prediction_count = cou[1:].sum().item()
    print('CHECK:',mae,acc,prediction_count,total_predicted,len(pred))
    if math.isnan(mae) or math.isnan(acc) or math.isnan(prediction_count):
        mae = 0
        acc = 0
        prediction_count = 0
        histogram = np.zeros(threshold).tolist
        return mae,acc,prediction_count,histogram,0
    return abs(mae),acc,prediction_count,list(histogram),total_predicted


study_name = [ 'BILSTM_2015',
                #'SMARTphone_abiad_2021',
                'LSTM_vandermeeren_2022',
                'Heuristic_Lee_2015',
                'CNN_BASIL_2020',
                #'ML_vandermeeren_2018'
            ]
root_dir = sys.path[0][:-4]

activity_map = {0:'standing',1:'walking',2:'stairs up',3:'stairs down',4:'elevator up',5:'elevator down'}
sensor_map = {0:'left',1:'right',2:'hand',3:'back'}
for study in study_name:
    print(f'\n{study}\n')
    folder = os.path.join(root_dir,study)
    pred = pd.read_csv(os.path.join(folder,'pred.csv'))
    label = pd.read_csv(os.path.join(folder,'label.csv'))[:1990800]
    print(pred.shape,label.shape)
    for i in range(4):
        idx = label.index[label['activity']==i]
        print(f'\nactivity: {activity_map[i]} ')
        p = torch.tensor(pred['start'][idx].to_numpy())
        l = torch.tensor(label['start'][idx].to_numpy())
        scores(p,l,30)

    for i in range(4):
        idx = label.index[label['sensor']==i]
        print(f'\nsensor {sensor_map[i]} ')
        p = torch.tensor(pred['start'][idx].to_numpy())
        l = torch.tensor(label['start'][idx].to_numpy())
        scores(p,l,30)

