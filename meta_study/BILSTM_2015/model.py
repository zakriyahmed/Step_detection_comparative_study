import torch
import torch.nn as nn

# Define the BiLSTM model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, 4)  # *2 for bidirectional
        #self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_size*2)
        out = self.fc(lstm_out)     # Apply FC layer to each time step, out shape: (batch_size, sequence_length, output_size)
        #out = self.softmax(out)     # Apply SoftMax to each time step
        return out

# Model parameters
#input_size = 6  # 3 accelerometer + 3 gyroscope values per time step
#hidden_size = 5  # As specified, each LSTM layer has 5 neurons
#output_size = 6  # Number of motion classes (forward walking, backward walking, etc.)

# Create the model
#model = BiLSTMModel(input_size, hidden_size, output_size)

# Print the model architecture
#print(model)

# Sample input for the model
#sample_input = torch.randn(128, 500, input_size)  # batch_size=128, sequence_length=500
#sample_output = model(sample_input)

#print(sample_output.shape)  # Should be [128, 500, output_size]
