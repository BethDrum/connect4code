import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Connect4(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6*7, 7),
            nn.ReLU(),
            nn.Linear(7, 12),
            nn.ReLU(),
            nn.Linear(12, 7),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    