from copy import deepcopy as dp
from scipy import signal as S # type: ignore
from torch.utils.data import Dataset
import pandas as pd # type: ignore
import os
import numpy as np


class BaseDataLoader():
    def __init__(self, root,ToFilter=False,AddMagnitude=True,AddGyro=True,normalize=True,relax=False,sensor = 'hand'):
        self.root = root
        self.individuals = os.listdir(self.root)
        self.ToFilter = ToFilter
        self.AddMagnitude = AddMagnitude
        self.AddGyro = AddGyro
        self.normalize = normalize
        self.relax = relax
        self.sos = S.butter(3, 3, 'lp', fs=80, output='sos')
        self.number_steps = 0
        self.sensor = sensor

        self.sensor_map = {'left_pocket':0,'right_pocket':1,'hand':2,'back_pocket':3,'all':0}

        
        if self.sensor == 'left_pocket':
            col_start = 3
            col_end = 4
        elif self.sensor == 'right_pocket':
            col_start = 4
            col_end = 5
        elif self.sensor == 'hand':
            col_start = 5
            col_end = 6
        elif self.sensor == 'back_pocket':
            col_start = 6
            col_end = 7
        elif self.sensor == 'all':
            col_start = 3
            col_end = 7
        if self.AddGyro:
            self.col_names = [[f' AccelX_{i}', f' AccelY_{i}', f' AccelZ_{i}', f' GyroX_{i}', f' GyroY_{i}', f' GyroZ_{i}'] for i in range(col_start,col_end)]
            self.col_names = [item for sublist in self.col_names for item in sublist]
        else:
            self.col_names = [[f' AccelX_{i}', f' AccelY_{i}', f' AccelZ_{i}'] for i in range(col_start,col_end)]
            self.col_names = [item for sublist in self.col_names for item in sublist]
        self.col_names.insert(0,' Activity')
        self.csv_files = []
        self.index_files = []
        for person in self.individuals:
            files = os.listdir(f'{self.root}/{person}')
            for file in files:
                if file.endswith('csv'):
                    self.csv_files.append(f'{self.root}/{person}/'+file)
                    self.index_files.append(f'{self.root}/{person}/'+file+'.stepNonoverlap')
        


        data,labels,sensor_labels,activity = self.read_sequence(0)
        for i in range(1,len(self.csv_files)):
            data_tmp,label_tmp,sensor_labels_tmp,activity_tmp = self.read_sequence(i)
            #print(sensor_labels_tmp.shape,sensor_labels.shape)
            data = np.concatenate((data, data_tmp), axis=0)
            labels = np.concatenate((labels,label_tmp),axis=0)
            sensor_labels = np.concatenate((sensor_labels,sensor_labels_tmp))
            activity = np.concatenate((activity,activity_tmp))


        #print(data.shape,labels.shape)
        self.data,self.labels = data,labels
        self.sensor_labels = sensor_labels
        self.activity = activity

    def read_sequence(self,idx):    
        csv = pd.read_csv(self.csv_files[idx],usecols=self.col_names)
        activity = csv.pop(' Activity')
        #csv = csv.to_numpy()
        if self.normalize==True:
            csv = (csv-csv.min())/(csv.max()-csv.min())

        labels = self.index_to_signal(self.index_files[idx],csv.shape[0])
        sensor_labels = np.zeros((labels.shape[0])) + self.sensor_map[self.sensor]
        if self.sensor == 'all':
            new_names=[' AccelX_', ' AccelY_',' AccelZ_', ' GyroX_', ' GyroY_', ' GyroZ_'] if self.AddGyro else [' AccelX_', ' AccelY_',' AccelZ_']
            new = pd.DataFrame()
            for col in new_names:
                col_to_merge = [col+'3', col+'4', col+'5', col+'6']
                new[col] = pd.concat([csv[col_to_merge[0]],
                                    csv[col_to_merge[1]],
                                    csv[col_to_merge[2]],
                                    csv[col_to_merge[3]]])
            labels = np.row_stack((labels,labels,labels,labels))
            sensor_labels = np.concatenate((sensor_labels+0,sensor_labels+1,sensor_labels+2,sensor_labels+3))
            activity = np.concatenate((activity,activity,activity,activity))
        else:
            new = dp(csv)

        if self.relax==True:
            onesS = np.where(labels[:,0]==1)[0]
            onesE = np.where(labels[:,1]==1)[0]
            all_indicesS = [(onesS+i).tolist() for i in range(-10,11)]
            all_indicesE = [(onesE+i).tolist() for i in range(-10,11)]
            #print(all_indices)
            labels[all_indicesS,0]=1
            labels[all_indicesE,1]=1

        new = new.to_numpy()

        if self.ToFilter:
            new = self.filter(new)

        if self.AddMagnitude:
            new = self.magnitude(new)

        return new,labels,sensor_labels,activity
        
        

    def index_to_signal(self,file,lenth):
        indexlabels = pd.read_csv(file,names=['start','end']).to_numpy()
        self.number_steps += len(indexlabels)
        zeros = np.zeros((lenth,2))
        #print(zeros.shape,indexlabels[-1,0],indexlabels[-1,1])
        zeros[[indexlabels[:,0]],0]=1
        zeros[[indexlabels[:,1]],1]=1

        return zeros
    
    def filter(self,array):
        
        filtered = S.sosfilt(self.sos, array.T)

        return np.array(filtered.T)
    
    def magnitude(self,array):
        if self.AddGyro:
            if array.shape[-1]<3:
                raise('Gyroscope data required but not loaded')
            
            magAccel = np.expand_dims(np.sqrt( array[:,0]**2 + array[:,1]**2 + array[:,2]**2),axis=1)#.unsqueeze(1)
            magGyro = np.expand_dims(np.sqrt( array[:,3]**2 + array[:,4]**2 + array[:,5]**2),axis=1)#.unsqueeze(1)
            #print(magAccel.shape,magGyro.shape,array.shape)
            mag = np.concatenate((array,magAccel,magGyro),1)
        
        else:
            mag = np.expand_dims(np.sqrt( array[:,0]**2 + array[:,1]**2 + array[:,2]**2),axis=1)#.unsqueeze(1)
            mag = np.concatenate((array,mag),1)

        return mag
    











