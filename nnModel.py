import torch
import torch.nn as nn
import torch.nn.functional as F


# Crate Model

# Input Layer --> 6 Feature
# Hidden Layer 1 --> 200 Neuron
# Hidden Layer 2 --> 200 Neuron
# Hidden Layer 2 --> 200 Neuron
# Output Dim --> 2


class DiabeticsNN(nn.Module):
    def __init__(self, num_features, num_classes, hiddenlayer1, hiddenlayer2, hiddenlayer3):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hiddenlayer1)
        self.fc2 = nn.Linear(hiddenlayer1, hiddenlayer2)
        self.fc3 = nn.Linear(hiddenlayer2, hiddenlayer3)
        self.out = nn.Linear(hiddenlayer3, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x

