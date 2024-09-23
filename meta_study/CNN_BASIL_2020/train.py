import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model import Conv1DModel as CNN # type: ignore
from dataloader import CNNDataloader
from test_models import Test


class Train():
    def __init__(self,root,device,learning_rate,epochs,model_name,individuals=None,test_individuals=[31]) -> None:
        self.root = root
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.window_size = 160
        self.stride = 1
        self.model_name = model_name
        self.model = CNN(window_size=self.window_size,total_features=3)
        self.individuals = individuals
        self.test_individuals = test_individuals
        self.dataset = CNNDataloader(self.root,
                                     ToFilter=True,
                                     AddMagnitude=False,
                                     AddGyro=False,
                                     normalize=True,
                                     window_size=self.window_size,
                                     stride=self.stride,
                                     windowed_labels=True,
                                     individuals=self.individuals)
        self.dataloader = DataLoader(self.dataset,batch_size=256,shuffle=True)
        self.loss_history_epoch = []
        self.acc_history_epoch = []
        files = os.listdir()
        if model_name in files:
            self.model = torch.load(self.model_name)

    

    def train(self):
        self.model = self.model.train().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        


        criterion = nn.MSELoss()

        try:
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_input,batch_label in self.dataloader:
                    
                    optimizer.zero_grad()
                    batch_input,batch_label = batch_input.to(self.device).float().permute(0,2,1),batch_label.to(self.device)

                    outputs = self.model(batch_input)
                    #print(batch_input.shape,outputs.shape,batch_label.shape)
                    loss = criterion(outputs[:,0],batch_label.float())

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss/len(self.dataloader)
                print(f'Epoch: {epoch},    Loss: {avg_loss}')
                self.loss_history_epoch.append(avg_loss)

                if epoch%10 == 0:
                    self.save_model(self.model_name)
                    b = Test(self.root,'cuda',self.model_name,during_training=True,individuals = self.test_individuals)
                    start,Tstart,_,_ = b.test()
                    acc = b.count_accuracy(start,Tstart)
                    self.acc_history_epoch.append(acc)
                    print(f'Epoch: {epoch},    Acc: {acc}')
            return self.model,self.loss_history_epoch
        except Exception as e:
            print(e)
            print('Interrupted. Saving model')
            np.save('loss.npy',np.array(self.loss_history_epoch))
            np.save('Acc.npy',np.array(self.acc_history_epoch))
            self.save_model(self.model_name)    

    def save_model(self,name):
        torch.save(self.model,name)
        

"""a = Train('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/train','cuda',0.001,21)
a.train()
"""