import numpy as np
import os
import sys
import pandas as pd # type: ignore
from torch.utils.data import Dataset
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from meta_study.dataloader.base_dataloader import BaseDataLoader


class LSTMDataloader(Dataset):

    def __init__(self,root,ToFilter=False,AddMagnitude=True,AddGyro=True,normalize=True,window_size=100,stride=1,windowed_labels=True,relaxed_labels=False) -> None:
        self.root = root
        self.ToFilter = ToFilter
        self.AddMagnitude = AddMagnitude
        self.AddGyro = AddGyro
        self.normalize = normalize
        self.window_size = window_size
        self.stride = stride
        self.windowed_labels  = windowed_labels
        self.relaxed_labels = relaxed_labels
        base_data = BaseDataLoader(self.root,self.ToFilter,self.AddMagnitude,self.AddGyro,self.normalize,self.relaxed_labels)

        data,labels = base_data.data,base_data.labels

        data = torch.tensor(data)
        labels = torch.tensor(labels)

        #print(data.shape,labels.shape)

        
        N = data.shape[0]
        if self.AddGyro:
            if self.AddMagnitude:
                channels = 8
                bp = int(data.shape[1]/(3+3+2))
            else:
                channels = 6
                bp = int(data.shape[1]/(3+3))
        else:
            if self.AddMagnitude:
                channels = 3+1
                bp = int(data.shape[1]/channels)
                #self.number_steps = base_data.number_steps*bp
            else:
                channels = 3
                bp = int(data.shape[1]/channels)
        data = data.view(N, bp, channels)
        data = data.permute(1, 0, 2).contiguous().view(N * bp, channels)


        labels = labels.repeat(bp, 1)
        num_windows = (len(data) - self.window_size) // self.stride + 1
        start_indices = [i * self.stride for i in range(num_windows)]
        self.windows_data = [data[i:i+self.window_size] for i in start_indices]


        self.windows_data = [data[i:i+self.window_size] for i in range(0,len(data)-self.window_size+1,self.stride)]
        if self.windowed_labels:
            self.windows_labels = [labels[i:i+self.window_size] for i in range(0,len(labels)-self.window_size+1,self.stride)]
            self.windows_labels = torch.stack(self.windows_labels)
        else:
            self.windows_labels = labels.clone()

        self.windows_data = torch.stack(self.windows_data)



    def __len__(self):
        return len(self.windows_data)

    def __getitem__(self, idx):
        if self.windowed_labels:
            return self.windows_data[idx],self.windows_labels[idx]
        else:
            return self.windows_data[idx] 