import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model import Conv1DModel as CNN # type: ignore
from dataloader import CNNDataloader


class Test():
    def __init__(self,root,device,learning_rate,epochs,during_training=False) -> None:
        self.root = root
        self.device = device
        self.epochs = epochs
        self.during_training = during_training
        self.learning_rate = learning_rate
        self.window_size = 160
        if self.during_training==True:
            self.stride = 160
        else:
            self.stride = 1
        self.model = CNN(window_size=self.window_size,total_features=3).to(self.device)
        self.dataset = CNNDataloader(self.root,ToFilter=True,AddMagnitude=False,AddGyro=False,normalize=True,window_size=self.window_size,stride=self.stride,windowed_labels=True)
        self.first_step = self.dataset.first_step
        self.dataloader = DataLoader(self.dataset,batch_size=1,shuffle=False)
        self.load()

    def load(self):
        self.model = torch.load(f"BASIL_2020_{self.epochs}_epochs.pt")

    def test(self):
        sigmoid = nn.Sigmoid()
        count = 0
        true_count = 0
        prev_pred_steps = 0
        pred_step_start_indices = []
        pred_step_end_indices = []
        #print(len(self.dataloader))
        for i, (test_input, test_label) in enumerate(self.dataloader):
            #print(test_input.shape)
            test_input = test_input.float().to(self.device).permute(0,2,1)
            
            output = self.model(test_input)
            count += (output[0] / self.window_size) * self.stride
            true_count += (test_label[0] / self.window_size) * self.stride
            if self.during_training == False:
                step_delta = int(count) - prev_pred_steps
                prev_pred_steps = int(count)

                if step_delta > 0:
                    for j in range(step_delta):
                        pred_step_start_indices.append(self.first_step - int(self.window_size/2) + self.stride* i)
                        pred_step_end_indices.append((self.first_step - int(self.window_size/2) + self.stride* i)-1)

            

        return count,true_count,pred_step_start_indices,pred_step_end_indices
    
    def count_accuracy(self,start_count,true_start_count):
        
        
        acc = start_count/true_start_count

        return acc*100
