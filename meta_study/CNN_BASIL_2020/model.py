import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DModel(nn.Module):
    def __init__(self, window_size, total_features):
        super(Conv1DModel, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=total_features, out_channels=10, kernel_size=10, stride=1)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=5)
        self.flatten = nn.Flatten()
        
        # The in_features for the Dense layer depends on the output size of the last conv layer
        # Calculate the output size after the convolutions
        conv_output_size = self._get_conv_output_size(window_size)
        self.fc1 = nn.Linear(conv_output_size, 1)

    def _get_conv_output_size(self, window_size):
        # Simulate a forward pass to determine the size of the flattened feature vector
        x = torch.zeros(1, self.conv1.in_channels, window_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return x.shape[1]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        return x

""" Example usage
window_size = 50  # Example window size
total_features = 3  # Example total features

model = Conv1DModel(window_size, total_features)
print(model)
"""