import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        
        # Fully connected layer to produce 2 outputs (start and end)
        self.fc = nn.Linear(hidden_size, 2)
        
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # (batch_size, sequence_length, hidden_size)
        
        # Apply the fully connected layer
        out = self.fc(out)  # (batch_size, sequence_length, 2)
        
        # Apply sigmoid activation to get probabilities for each output
        #out = self.sigmoid(out)  # (batch_size, sequence_length, 2)
        
        # Separate the outputs for start and end
        start_output = out[:, :, 0]  # (batch_size, sequence_length)
        end_output = out[:, :, 1]    # (batch_size, sequence_length)
        
        return start_output, end_output
    
    def zeros(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        return h0,c0
    
    def testing(self,x,h0,c0):
        out, (h0,c0) = self.lstm(x,(h0,c0))
        out = self.fc(out)
        start_output = out[:, :, 0]  # (batch_size, sequence_length)
        end_output = out[:, :, 1]    # (batch_size, sequence_length)
        
        return start_output, end_output, (h0,c0)

#model = LSTM(3,400,2,0.2)
#input = torch.zeros((10,200,3))
#x,y = model(input)
#print(x.shape)