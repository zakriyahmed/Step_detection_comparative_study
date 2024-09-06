import os
import traceback


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from loss import BCEWithWeights
from model import BiLSTMModel as LSTM
from dataloader import LSTMDataloader


class Train():
    def __init__(self,root,device,learning_rate,epochs) -> None:
        self.root = root
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.ws0,self.we0,self.ws1,self.we1 = 1,1,1,1
        self.model = LSTM(input_size=6, hidden_size=5, num_layers=1)
        self.dataset = LSTMDataloader(self.root,
                                      ToFilter=False,
                                      AddMagnitude=False,
                                      AddGyro=True,
                                      normalize=True,
                                      window_size=500,
                                      stride=100,
                                      windowed_labels=True)
        self.dataloader = DataLoader(self.dataset,batch_size=128,shuffle=True)
        self.loss_history_epoch = []
        self.acc_history_epoch = []
        files = os.listdir()
        if f'BILSTM_2015_{self.epochs}_epochs.pt' in files:
            self.model = torch.load(f'BILSTM_2015_{self.epochs}_epochs.pt')

    def train(self):
        print("training started\n")
        self.model = self.model.train().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate,betas=(0.01,0.01))
        


        criterion = BCEWithWeights(self.ws0,self.we0,self.ws1,self.we1,self.device)

        try:
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_input,batch_label in self.dataloader:
                    optimizer.zero_grad()
                    batch_input,batch_label = batch_input.to(self.device).float(),batch_label.to(self.device)

                    outputs = self.model(batch_input)
                    #print(batch_input.shape,outputs.shape,batch_label.shape)
                    loss = criterion(outputs,batch_label)

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss/len(self.dataloader)
                print(f'Epoch: {epoch},    Loss: {avg_loss}')
                self.loss_history_epoch.append(avg_loss)

                if epoch%10 == 0:
                    self.save_model()
                    b = Test('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/test','cuda',0.001,self.epochs,'all')
                    start,_ = b.test()
                    acc = b.count_accuracy(start,torch.tensor(b.ground_labels))
                    self.acc_history_epoch.append(acc)
                    print(f'Epoch: {epoch},    Acc: {acc}')
            return self.model,self.loss_history_epoch
        except Exception as e:
            print(e)
            traceback.print_exc()
            print('Interrupted. Saving model')
            np.save('loss.npy',np.array(self.loss_history_epoch))
            np.save('Acc.npy',np.array(self.acc_history_epoch))
            self.save_model()


    def save_model(self):
        torch.save(self.model,f"BILSTM_2015_{self.epochs}_epochs.pt")
        

class Test():
    def __init__(self,root,device,learning_rate,epochs,sensor) -> None:
        self.root = root
        self.window_size = 500
        self.device = device
        self.epochs = epochs
        self.sensor = sensor
        self.learning_rate = learning_rate
        self.ws0,self.we0,self.ws1,self.we1 = 0.739,0.739,1.548,1.548
        self.model = LSTM(input_size=6, hidden_size=5, num_layers=1).to(self.device)
        self.dataset = LSTMDataloader(self.root,
                                      ToFilter=False,
                                      AddMagnitude=False,
                                      AddGyro=True,
                                      normalize=True,
                                      window_size=self.window_size,
                                      stride=1,
                                      windowed_labels=True,
                                      sensor=self.sensor)
        self.ground_labels = self.dataset.ground_labels
        self.activity = self.dataset.activity
        self.sensor_labels = self.dataset.sensor_labels
        
        self.dataloader = DataLoader(self.dataset,batch_size=1024,shuffle=False)
        self.load()

    def load(self):
        self.model = torch.load(f"BILSTM_2015_{self.epochs}_epochs.pt")

    def test(self):
        sigmoid = nn.Sigmoid()
        start = True
        #print(len(self.dataloader))
        for test_input,test_label in self.dataloader:
            if start:
                #print(test_input.shape)
                test_input = test_input.float().to(self.device)
                
                output = self.model(test_input)
                
                logits_S,logits_E = output[:,:,0],output[:,:,1]
                logits_S, logits_E = sigmoid(logits_S),sigmoid(logits_E)
                #print(output.shape,logits_E.shape)
                logits_S[logits_S>=0.5] = 1
                logits_S[logits_S<0.5] = 0

                logits_E[logits_E>=0.5] = 1
                logits_E[logits_E<0.5] = 0

                start = False

                #start_preds = logits_S.clone().cpu().detach()
                #end_preds = logits_E.clone().cpu().detach()
                #all_labels = test_label.clone().cpu().detach()
                #print(logits_E.shape,logits_S.shape)
                #non_overlapping_labels = self.diag_sum(all_labels[:,0],all_labels[:,1])
                votedPred = self.diag_sum(logits_S,logits_E).cpu()
                asc = torch.stack((torch.arange(1,self.window_size,1),torch.arange(1,self.window_size,1))).T
                des = torch.stack((torch.arange(self.window_size-1,0,-1),torch.arange(self.window_size-1,0,-1))).T

            else:
                #print(test_input.shape)
                test_input = test_input.float().to(self.device)
                
                output = self.model(test_input)
                logits_S,logits_E = output[:,:,0],output[:,:,1]
                logits_S, logits_E = sigmoid(logits_S),sigmoid(logits_E)
                #print(output.shape,logits_E.shape)
                logits_S[logits_S>=0.5] = 1
                logits_S[logits_S<0.5] = 0

                logits_E[logits_E>=0.5] = 1
                logits_E[logits_E<0.5] = 0
                #start_preds = torch.cat((start_preds,logits_S.clone().cpu().detach()),dim=0)
                #end_preds = torch.cat((end_preds,logits_E.clone().cpu().detach()),dim=0)
                #all_labels_tmp = test_label.clone().cpu().detach()
                #non_overlapping_labels_tmp = self.diag_sum(all_labels_tmp[:,0],all_labels_tmp[:,1])

                #mid_term_label = ((non_overlapping_labels[-self.window_size+1:]*des) + (non_overlapping_labels_tmp[:self.window_size-1]*asc))/self.window_size
                
                #non_overlapping_labels = torch.cat((non_overlapping_labels[:-self.window_size+1],mid_term_label,non_overlapping_labels_tmp[self.window_size-1:]),dim=0) 

                votedPred_tmp = self.diag_sum(logits_S,logits_E).cpu()
                mid_term = ((votedPred[-self.window_size+1:]*des) + (votedPred_tmp[:self.window_size-1]*asc))/self.window_size

                votedPred = torch.cat((votedPred[:-self.window_size+1],mid_term,votedPred_tmp[self.window_size-1:]),dim=0) 

        return votedPred[:,0],votedPred[:,1]
    

    def diag_sum(self,tensorS,tensorE):
        #
        #      Find the reverse diagonals sum and average it by total number of element in diagonal
        #
        #      [1, 2, 3]                                                                                          [1, 2, 3]
        #      [4, 5, 6]  --Diagonals--> [[1],[2,4],[3,5,7],[6,8],[9]]  --Average--> [[1],[3],[5],[7],[9]]   ==     +[4, 5, 6]
        #      [7, 8, 9]                                                                                               +[7, 8, 9]
        #                                                                                                          ---------------
        #                                                                                                          [1,6 ,15, 14,9]/[1,2,3,2,1]
        #                                                                                                          ----------------
        #                                                                                                        = [1,3, 5, 7 ,9] 
        #      It first flip the tensor to get the reverse diagonals then find average values and flip the result back.
        #      We do not need to take into account any window_size here, as it is divided by number of values in a diagonal  
        #
        tensorS = torch.flip(tensorS, dims=(1,))
        tensorE = torch.flip(tensorE,dims = (1,))
        votedPredS = []
        votedPredE = []
        for i in range(-tensorS.shape[0]+1,tensorS.shape[1]):
            diagS = torch.diag(tensorS,i)
            diagE = torch.diag(tensorE,i)
            valueS = diagS.sum()/diagS.shape[0]
            valueE = diagE.sum()/diagE.shape[0]
            votedPredS.append(valueS.item())
            votedPredE.append(valueE.item())
        votedPredE = torch.tensor(votedPredE)
        votedPredE = torch.flip(votedPredE,dims=(0,))

        votedPredS = torch.tensor(votedPredS)
        votedPredS = torch.flip(votedPredS,dims=(0,))

        return torch.stack((votedPredS,votedPredE),dim=1)
    
    def count_accuracy(self,start_preds,all_labels):
        start_count = torch.count_nonzero(start_preds)
        true_start_count = torch.count_nonzero(all_labels[:,0])
        
        acc = start_count/true_start_count

        return acc*100

#b = Train('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/train','cuda',0.001,30)
#b.train()
#b.save_model()
#a = Test('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/test','cuda',0.001,30)
#x_,y_ = a.test()
#print(x_.shape,y_.shape)

            