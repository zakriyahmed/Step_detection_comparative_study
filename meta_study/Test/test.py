import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import pandas as pd # type: ignore 
import torch
import numpy as np

study_name = [ 'BILSTM_2015',
                #'SMARTphone_abiad_2021',
                #'LSTM_vandermeeren_2022',
                'Heuristic_Lee_2015',
                'CNN_BASIL_2020',
                'ML_vandermeeren_2018'
            ]
root_dir = sys.path[0][:-4]

for study in study_name:
    folder = os.path.join(root_dir,study)
    pred = pd.read_csv(os.path.join(folder,'pred.csv'))
    print(pred.shape)