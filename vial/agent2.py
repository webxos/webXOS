import torch
import torch.nn as nn

class Agent2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# Version 1.0 - Baseline template for vial2 coordination
# Updated on August 10, 2025
