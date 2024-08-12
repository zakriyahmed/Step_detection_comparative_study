from copy import deepcopy as dp
from scipy import signal as S # type: ignore
from torch.utils.data import Dataset
import pandas as pd # type: ignore
import os
import numpy as np


class BaseDataLoader():
    def __init__(self, root,ToFilter=False,AddMagnitude=True,AddGyro=True,normalize=True,relax=False):
        self.root = root
        self.individuals = os.listdir(self.root)
        self.ToFilter = ToFilter
        self.AddMagnitude = AddMagnitude
        self.AddGyro = AddGyro
        self.normalize = normalize
        self.relax = relax
        self.sos = S.butter(3, 3, 'lp', fs=80, output='sos')
        self.number_steps = 0
        if self.AddGyro:
            self.col_names = [[f' AccelX_{i}', f' AccelY_{i}', f' AccelZ_{i}', f' GyroX_{i}', f' GyroY_{i}', f' GyroZ_{i}'] for i in range(5,6)]
            self.col_names = [item for sublist in self.col_names for item in sublist]
        else:
            self.col_names = [[f' AccelX_{i}', f' AccelY_{i}', f' AccelZ_{i}'] for i in range(5,6)]
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
        
        try:
            self.maximums=np.load('maximums.npy')
            self.minimums=np.load('minimums.npy')
            self.total_signal_length = np.load('total_signal_length.npy')[0]
            self.max_min = self.maximums-self.minimums
        except:
            self.stats()

        data,labels = self.read_sequence(0)
        for i in range(1,len(self.csv_files)):
            data_tmp,label_tmp = self.read_sequence(i)

            data = np.concatenate((data, data_tmp), axis=0)
            labels = np.concatenate((labels,label_tmp),axis=0)


        #print(data.shape,labels.shape)
        self.data,self.labels = data,labels


    def read_sequence(self,idx):    
        csv = pd.read_csv(self.csv_files[idx],usecols=self.col_names)
        self.activity = csv.pop(' Activity')
        csv = csv.to_numpy()
        labels = self.index_to_signal(self.index_files[idx],csv.shape[0])
        if self.relax==True:
            onesS = np.where(labels[:,0]==1)[0]
            onesE = np.where(labels[:,1]==1)[0]
            all_indicesS = [(onesS+i).tolist() for i in range(-10,11)]
            all_indicesE = [(onesE+i).tolist() for i in range(-10,11)]
            #print(all_indices)
            labels[all_indicesS,0]=1
            labels[all_indicesE,1]=1

        if self.ToFilter:
            csv = self.filter(csv)
        if self.normalize==True:
            #print(data.shape,self.max_min.shape)
            csv = (csv-self.minimums[1:])/self.max_min[1:]
            #print('normalized')
        if self.AddMagnitude:
            csv = self.magnitude(csv)

        return csv,labels
        
    def stats(self):    
        if self.normalize:
            self.maximums = []
            self.minimums = []
            self.total_signal_length = 0
            for i in self.csv_files:
                data = pd.read_csv(i,usecols=self.col_names).to_numpy()
                self.total_signal_length =+ data.shape[0]
                self.maximums.append(data.max(axis=0))
                self.minimums.append(data.min(axis=0))
            self.maximums = np.max(np.array(self.maximums),axis=0)
            self.minimums = np.min(np.array(self.minimums),axis=0)
            np.save('maximums.npy',self.maximums)
            np.save('minimums.npy',self.minimums)
            np.save('total_signal_length.npy',np.array([self.total_signal_length]))
            del data,i
            self.max_min = self.maximums-self.minimums
        

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
    











