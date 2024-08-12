import os
import sys
import pickle

import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore
from sklearn.utils.class_weight import compute_class_weight # type: ignore


from feature_extraction import * # type: ignore
from dataloader import Dataloader


root = '/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/test'
report = pd.read_csv('grid_search.csv')
ToFilter = True
AddMagnitude = True
AddGyro = False
normalize = True
idx = report['accuracy'].idxmax()
window_size = report['ws'][idx]
stride = window_size
a = Dataloader(root,ToFilter=ToFilter,AddMagnitude=AddMagnitude,AddGyro=AddGyro,normalize=normalize, windowed_labels=True)
windows,labels = a.getdata(window_size,stride)

data,label = all_features(windows,labels) # type: ignore

with open('svm_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

pred = loaded_model.predict(data)

pred_windows = np.zeros_like(labels)

for value,i in zip(pred,range(len(pred))):
    #print(value,i)
    for j in range(1,np.unique(pred)[-1]+1):
        if value==2:
            print(value,i,j)
        if value==j:
            for k in range(value):
                if value==2:
                    print(value,i,j,k)
                pred_windows[i, (int(window_size/(j+1))*(k+1))-1,0] = 1
                pred_windows[i, (int(window_size/(j+1))*(k+1))-2,1] = 1


"""    if value ==1:
        pred_windows[i,int(window_size/2)-1,0] = 1
        pred_windows[i,int(window_size/2)-2,1] = 1

    if value ==2:
        pred_windows[i,int(window_size/3)-1,0] = 1
        pred_windows[i,int(window_size/3)-2,1] = 1

        pred_windows[i,(int(window_size/3)-1)*2,0] = 1
        pred_windows[i,(int(window_size/3)-2)*2,1] = 1"""

    
prediction = pred_windows.reshape(-1,2)
df = {'start':prediction[:,0],'end':prediction[:,1]}
df = pd.DataFrame(df)
df.to_csv('prediction.csv')