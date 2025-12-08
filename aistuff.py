import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium

from gymnasium import Env, spaces
board = [6][7]

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
    

class C4(Env, board):
    def __init__(self):
        super(C4, self).__init__()

        self.observation_space = board
        self.action_space #output of NN, picking the columns to play in
    
    def reset(self):
        self.ep_return = 0
        for i in range (6):
            for j in range (7):
                board[i][j] = 0
        return super().reset()
    
    def step(self, action):
        done = False
        reward = 1
        #if board == full or win - reset board, done = true
        #run nn and place a token
        self.ep_return += 1
        return super().step()