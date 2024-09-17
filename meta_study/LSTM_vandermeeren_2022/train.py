import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from loss import BCEWithWeights
from model import LSTM
from dataloader import LSTMDataloader


class Train():
    def __init__(self,root,device,learning_rate,epochs) -> None:
        self.root = root
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        #self.ws0,self.we0,self.ws1,self.we1 = 0.739,0.739,1.548,1.548
        self.model = LSTM(input_size=4, hidden_size=400, num_layers=2)
        self.dataset = LSTMDataloader(self.root,ToFilter=True,AddMagnitude=True,AddGyro=False,normalize=True,window_size=200,stride=40,windowed_labels=True,relaxed_labels=True)
        self.dataloader = DataLoader(self.dataset,batch_size=1024,shuffle=True)
        self.ground_labels = self.dataset.ground_labels
        self.n_ones = np.count_nonzero(self.ground_labels[:,0]) 
        #print(self.n_ones,self.ground_labels.shape)
        self.n_zeros = self.ground_labels[:,0].shape[0] - self.n_ones
        self.ws0 = (self.n_ones + self.n_zeros) / (2*self.n_zeros)
        self.ws1 = (self.n_ones + self.n_zeros) / (2*self.n_ones)
        self.we0 = self.ws0
        self.we1 = self.ws1
        self.loss_history_epoch = []

    def train(self):
        self.model = self.model.train().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        


        criterion = BCEWithWeights(self.ws0,self.we0,self.ws1,self.we1,self.device)

        try:
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_input,batch_label in self.dataloader:
                    
                    optimizer.zero_grad()
                    batch_input,batch_label = batch_input.to(self.device).float(),batch_label.to(self.device)

                    outputs = self.model(batch_input)
                    #print(outputs[0].shape,batch_label.shape)
                    loss = criterion(outputs,batch_label)

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss/len(self.dataloader)
                print(f'Epoch: {epoch},    Loss: {avg_loss}')
                self.loss_history_epoch.append(avg_loss)
        except Exception as e:
            print("Interupted")
            self.save_model()

        return self.model,self.loss_history_epoch


    def save_model(self):
        torch.save(self.model,f"LSTM_vandermeeren_{self.epochs}_epochs.pt")
        

class Test():
    def __init__(self,root,device,learning_rate,epochs) -> None:
        self.root = root
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.ws0,self.we0,self.ws1,self.we1 = 0.739,0.739,1.548,1.548
        self.model = LSTM(input_size=4, hidden_size=400, num_layers=2).to(self.device)
        self.dataset = LSTMDataloader(self.root,ToFilter=True,AddMagnitude=True,AddGyro=False,normalize=True,window_size=200,stride=200,windowed_labels=True)
        self.dataloader = DataLoader(self.dataset,batch_size=1,shuffle=False)
        self.ground_labels = self.dataset.ground_labels
        self.activity = self.dataset.activity
        self.sensor_labels = self.dataset.sensor_labels
        
        self.load()

    def load(self):
        self.model = torch.load(f"LSTM_vandermeeren_{self.epochs}_epochs.pt").to(self.device)

    def test(self):
        sigmoid = nn.Sigmoid()
        start = True
        for test_input,test_label in self.dataloader:
            if start:
                #print(test_input.shape)
                test_input = test_input.float().to(self.device)
                #h0,c0 = self.model.zeros(test_input)
                logits_S,logits_E = self.model(test_input)
                logits_S, logits_E = sigmoid(logits_S),sigmoid(logits_E)

                logits_S[logits_S>=0.5] = 1
                logits_S[logits_S<0.5] = 0

                logits_E[logits_E>=0.5] = 1
                logits_E[logits_E<0.5] = 0

                start = False

                start_preds = logits_S.clone().cpu().detach()
                end_preds = logits_E.clone().cpu().detach()
                all_labels = test_label.clone().cpu().detach()

            else:
                #print(test_input.shape)
                test_input = test_input.float().to(self.device)
                
                logits_S,logits_E = self.model(test_input)
                logits_S, logits_E = sigmoid(logits_S),sigmoid(logits_E)

                logits_S[logits_S>=0.5] = 1
                logits_S[logits_S<0.5] = 0

                logits_E[logits_E>=0.5] = 1
                logits_E[logits_E<0.5] = 0
                #print(logits_S)
                #print(start_preds.shape,logits_S.shape)
                start_preds = torch.cat((start_preds,logits_S.clone().cpu().detach()),dim=1)
                end_preds = torch.cat((end_preds,logits_E.clone().cpu().detach()),dim=1)
                all_labels = torch.cat((all_labels,test_label.clone().cpu().detach()),dim=1)


        return start_preds,end_preds
    
#b = Train('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/train','cuda',0.001,2)
#b.train()
#b.save_model()
#a = Test('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/test','cuda',0.001,30)
#x_,y_ = a.test()
#print(x_.shape,y_.shape)

            