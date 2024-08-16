import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model import Conv1DModel as CNN # type: ignore
from dataloader import CNNDataloader


class Test():
    def __init__(self,root,device,learning_rate,epochs) -> None:
        self.root = root
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.window_size = 160
        self.stride = 160
        self.model = CNN(window_size=self.window_size,total_features=3).to(self.device)
        self.dataset = CNNDataloader(self.root,ToFilter=True,AddMagnitude=False,AddGyro=False,normalize=True,window_size=self.window_size,stride=self.stride,windowed_labels=True)
        self.dataloader = DataLoader(self.dataset,batch_size=1024,shuffle=False)
        self.load()

    def load(self):
        self.model = torch.load(f"BASIL_2020_{self.epochs}_epochs.pt")

    def test(self):
        sigmoid = nn.Sigmoid()
        count = 0
        true_count = 0
        #print(len(self.dataloader))
        for test_input,test_label in self.dataloader:
            #print(test_input.shape)
            test_input = test_input.float().to(self.device).permute(0,2,1)
            
            output = self.model(test_input)
            count += output[0]
            true_count += test_label[0]

            
                

        return count,true_count
    
    def count_accuracy(self,start_count,true_start_count):
        
        
        acc = start_count/true_start_count

        return acc*100
