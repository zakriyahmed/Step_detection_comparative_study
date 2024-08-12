import torch
import torch.nn as nn

class BCEWithWeights(nn.Module):
    def __init__(self,ws0,we0,ws1,we1,device='cuda'):
        super(BCEWithWeights,self).__init__()
        self.ws0 = ws0/(ws0+ws1)
        self.we0 = we0/(we0+we1)
        self.ws1 = ws1/(ws0+ws1)
        self.we1 = we1/(we0+we1)
        self.device = device
        
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self,outputs,targets):
        #print(outputs[0].shape, targets[:,:,0].shape)
        loss_s = self.criterion(outputs[:,:,0], targets[:,:,0])
        loss_e = self.criterion(outputs[:,:,1], targets[:,:,1])

        w1 = torch.ones((targets.shape[0],targets.shape[1]),device=self.device)*self.ws0
        w1[[targets[:,:,0]==1]]=self.ws1

        w2 = torch.ones((targets.shape[0],targets.shape[1]),device=self.device)*self.we0
        w2[[targets[:,:,1]==1]]=self.we1

        loss= (torch.mean(loss_s*w1) + torch.mean(loss_e*w2))/2

        return loss
    

#l = BCEWithWeights(0.7,0.7,1.5,1.5)
#x = torch.zeros((10,200))
#y = torch.zeros((10,200))
#print(l(x,y))