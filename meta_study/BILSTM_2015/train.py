import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


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
        self.model = LSTM(input_size=4, hidden_size=5, num_layers=1)
        self.dataset = LSTMDataloader(self.root,ToFilter=True,AddMagnitude=True,AddGyro=False,normalize=True,window_size=500,stride=1,windowed_labels=True)
        self.dataloader = DataLoader(self.dataset,batch_size=1024,shuffle=True)
        self.loss_history_epoch = []

    def train(self):
        self.model = self.model.train().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        


        criterion = BCEWithWeights(self.ws0,self.we0,self.ws1,self.we1,self.device)


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

        return self.model,self.loss_history_epoch


    def save_model(self):
        torch.save(self.model,f"BILSTM_2015_{self.epochs}_epochs.pt")
        

class Test():
    def __init__(self,root,device,learning_rate,epochs) -> None:
        self.root = root
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.ws0,self.we0,self.ws1,self.we1 = 0.739,0.739,1.548,1.548
        self.model = LSTM(input_size=4, hidden_size=5, num_layers=1).to(self.device)
        self.dataset = LSTMDataloader(self.root,ToFilter=True,AddMagnitude=True,AddGyro=False,normalize=True,window_size=500,stride=500,windowed_labels=True)
        self.dataloader = DataLoader(self.dataset,batch_size=1,shuffle=False)
        self.load()

    def load(self):
        self.model = torch.load(f"BILSTM_2015_{self.epochs}_epochs.pt")

    def test(self):
        sigmoid = nn.Sigmoid()
        start = True
        print(len(self.dataloader))
        for test_input,test_label in self.dataloader:
            if start:
                #print(test_input.shape)
                test_input = test_input.float().to(self.device)
                
                output = self.model(test_input)
                
                logits_S,logits_E = output[0,:,0],output[0,:,1]
                logits_S, logits_E = sigmoid(logits_S),sigmoid(logits_E)
                #print(output.shape,logits_E.shape)
                logits_S[logits_S>=0.8] == 1
                logits_S[logits_S<0.8] == 0

                logits_E[logits_E>=0.8] == 1
                logits_E[logits_E<0.8] == 0

                start = False

                start_preds = logits_S.clone().cpu().detach()
                end_preds = logits_E.clone().cpu().detach()
                all_labels = test_label.clone().cpu().detach()

            else:
                #print(test_input.shape)
                test_input = test_input.float().to(self.device)
                
                output = self.model(test_input)
                logits_S,logits_E = output[0,:,0],output[0,:,1]
                logits_S, logits_E = sigmoid(logits_S),sigmoid(logits_E)
                #print(output.shape,logits_E.shape)
                logits_S[logits_S>=0.8] == 1
                logits_S[logits_S<0.8] == 0

                logits_E[logits_E>=0.8] == 1
                logits_E[logits_E<0.8] == 0
                
                start_preds = torch.cat((start_preds,logits_S.clone().cpu().detach()),dim=0)
                end_preds = torch.cat((end_preds,logits_E.clone().cpu().detach()),dim=0)
                all_labels = torch.cat((all_labels,test_label.clone().cpu().detach()),dim=0)


        return start_preds,end_preds
    
#b = Train('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/train','cuda',0.001,30)
#b.train()
#b.save_model()
#a = Test('/home/ann_ss22_group4/step detection/SIMUL-dataset/data/by-person/test','cuda',0.001,30)
#x_,y_ = a.test()
#print(x_.shape,y_.shape)

            