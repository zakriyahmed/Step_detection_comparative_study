
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




def grid_search():
    window_sizes = [30,50,70,90,110,130]
    cs = [0.1,0.5,1,1.5,2,3,4,5,6]
    gammas = ['scale','auto',0.1,1,5,10]
    report = {'ws':[],'gamma':[],'c':[],'accuracy':[]}

    root = '/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/train'
    ToFilter = True
    AddMagnitude = True
    AddGyro = False
    normalize = True
    a = Dataloader(root,ToFilter=ToFilter,AddMagnitude=AddMagnitude,AddGyro=AddGyro,normalize=normalize, windowed_labels=True)


    for window_size in window_sizes:
        stride = int(window_size)
        windows,labels = a.getdata(window_size,stride)

        data,label = all_features(windows,labels) # type: ignore 

        class_weights = compute_class_weight('balanced', classes=np.unique(label), y=label)
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
        for c in cs:
            for gamma in gammas:
                print(f'C:{c} \tgamma:{gamma} \t ws:{window_size}')
                

                rbf_svm = SVC(kernel='rbf', C=c, gamma=gamma,class_weight=class_weights_dict, random_state=42)
                rbf_svm.fit(X_train, y_train)

                y_pred = rbf_svm.predict(X_test)
                # Evaluate the model
                accuracy = accuracy_score(y_test, y_pred)

                report['accuracy'].append(accuracy)
                report['c'].append(c)
                report['gamma'].append(gamma)
                report['ws'].append(window_size)
                print( f'\t\t\t\t accuracy:{accuracy}')

                report_df = pd.DataFrame(report)
                report_df.to_csv('grid_search_all.csv')

def save_best_model():
    root = '/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/train'
    report = pd.read_csv('grid_search_all.csv')
    ToFilter = True
    AddMagnitude = True
    AddGyro = False
    normalize = True
    idx = report['accuracy'].idxmax()
    c = report['c'][idx]
    gamma = report['gamma'][idx]
    if gamma !='auto' and gamma != 'scale':
        gamma = int(gamma)
    window_size = report['ws'][idx]
    stride = window_size
    a = Dataloader(root,ToFilter=ToFilter,AddMagnitude=AddMagnitude,AddGyro=AddGyro,normalize=normalize, windowed_labels=True)
    windows,labels = a.getdata(window_size,stride)

    data,label = all_features(windows,labels) # type: ignore

    class_weights = compute_class_weight('balanced', classes=np.unique(label), y=label)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

    rbf_svm = SVC(kernel='rbf', C=c, gamma=gamma,class_weight=class_weights_dict, random_state=42)
    rbf_svm.fit(X_train, y_train)

    with open('svm_model_all.pkl', 'wb') as file:
        pickle.dump(rbf_svm, file)



if __name__ == '__main__':
    grid_search()
    save_best_model()



                        

