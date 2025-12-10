import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.spaces import discrete
from random import randint
from gymnasium import Env, spaces
import pygame

pygame.init()
SCREEN_WIDTH = 840
SCREEN_HEIGHT = 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Connect4")
pygame.display.update()




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
    




def checkWin(board, piece):
    for i in range(6):
        for j in range(7):
            if (j <= 3 and board[i][j] == piece and board[i][j+1] == piece and board[i][j+2] == piece and board[i][j+3] == piece):
                return True
            #vertical
            if (i <= 2 and board[i][j] == piece and board[i+1][j] == piece and board[i+2][j] == piece and board[i+3][j] == piece):
                return True
            #diagonal right 
            if (i <= 2 and j <= 3 and board[i][j] == piece and board[i+1][j+1] == piece and board[i+2][j+2] == piece and board[i+3][j+3] == piece):
                return True
            #diagonal left 
            if ( i >= 3 and j <= 3 and board[i][j] == piece and board[i-1][j+1] == piece and board[i-2][j+2] == piece and board[i-3][j+3] == piece):
                return True
    return False

def checkFull(board):
    for i in range (7):
        if board [0][i] != 0:
            return True
    return False    

def placeToken(board, j, piece):
    for i in reversed(range(6)):
        if board[i][j] == 0:
            board[i][j] = piece
            return board
    return board

#               PLEASE CALL THESE FUNCTIONS /BEFORE/ CHECKWIN. IF CHECKWIN IS CALLED BEFORE, NO REWARDS ARE AWARDED FOR WINNING

# function calculate rewards gathered from 4's, 3's, 2's, 1's in a row
def rewardRows(board, piece):
    for i in range(6):
        for j in range(7):
            # /////     horizontal     /////
            # 4 in a row
            if (j <= 3 and board[i][j] == piece and board[i][j+1] == piece and board[i][j+2] == piece and board[i][j+3] == piece):
                return 64 # reward = 64
            # 3 in a row
            elif (j <= 3 and board[i][j] == piece and board[i][j+1] == piece and board[i][j+2] == piece):
                return 8 # reward = 8
            # 2 in a row
            elif (j <= 3 and board[i][j] == piece and board[i][j+1] == piece):
                return 2 # reward # 2

            # /////     vertical    /////
            # 4 in a row
            if (i <= 2 and board[i][j] == piece and board[i+1][j] == piece and board[i+2][j] == piece and board[i+3][j] == piece):
                return 64
            # 3 in a row
            elif (i <= 2 and board[i][j] == piece and board[i+1][j] == piece and board[i+2][j] == piece):
                return 8
            # 2 in a row
            elif (i <= 2 and board[i][j] == piece and board[i+1][j] == piece):
                return 2

            # /////     diagonal right     /////
            # 4 in a row
            if (i <= 2 and j <= 3 and board[i][j] == piece and board[i+1][j+1] == piece and board[i+2][j+2] == piece and board[i+3][j+3] == piece):
                return 64
            # 3 in a row
            elif (i <= 2 and j <= 3 and board[i][j] == piece and board[i+1][j+1] == piece and board[i+2][j+2] == piece):
                return 8
            # 2 in a row
            elif (i <= 2 and j <= 3 and board[i][j] == piece and board[i+1][j+1] == piece):
                return 2
            
            # /////    diagonal left   /////
            # 4 in a row
            if ( i >= 3 and j <= 4 and board[i][j] == piece and board[i-1][j+1] == piece and board[i-2][j+2] == piece and board[i-3][j+3] == piece):
                return 64
            # 3 in a row
            elif ( i >= 3 and j <= 4 and board[i][j] == piece and board[i-1][j+1] == piece and board[i-2][j+2] == piece):
                return 8
            # 2 in a row
            elif ( i >= 3 and j <= 4 and board[i][j] == piece and board[i-1][j+1] == piece):
                return 2

    return 1 # generic 'place token' reward

# function to calculate rewards gathered from blocking opponent plays
def rewardOpponentBlock(board, myPiece, otherPiece):
    for i in range (6):
        for j in range (7):
            # /////     horizontal     /////
            # blocking a 3 in a row
            if (j <= 3 and board[i][j] == myPiece and board[i][j+1] == otherPiece and board[i][j+2] == otherPiece and board[i][j+3] == otherPiece):
                return 8 # reward = 8
            elif (j <= 3 and board[i][j] == otherPiece and board[i][j+1] == myPiece and board[i][j+2] == otherPiece and board[i][j+3] == otherPiece):
                return 8 
            elif (j <= 3 and board[i][j] == otherPiece and board[i][j+1] == otherPiece and board[i][j+2] == myPiece and board[i][j+3] == otherPiece):
                return 8    
            elif (j <= 3 and board[i][j] == otherPiece and board[i][j+1] == otherPiece and board[i][j+2] == otherPiece and board[i][j+3] == myPiece):
                return 8 

            # blocking a 2 in a row
            if (j <= 3 and board[i][j] == myPiece and board[i][j+1] == otherPiece and board[i][j+2] == otherPiece):
                return 4 # reward = 4
            elif (j <= 3 and board[i][j] == otherPiece and board[i][j+1] == myPiece and board[i][j+2] == otherPiece):
                return 4 
            elif (j <= 3 and board[i][j] == otherPiece and board[i][j+1] == otherPiece and board[i][j+2] == myPiece):
                return 4 

            # blocking a 1 in a row
            if (j <= 3 and board[i][j] == myPiece and board[i][j+1] == otherPiece):
                return 2 # reward # 2
            elif (j <= 3 and board[i][j] == otherPiece and board[i][j+1] == myPiece):
                return 2 # reward # 2


            # /////     vertical    /////
            # blocking 4 in a row
            if (i <= 2 and board[i][j] == myPiece and board[i+1][j] == otherPiece and board[i+2][j] == otherPiece and board[i+3][j] == otherPiece):
                return 8
            elif (i <= 2 and board[i][j] == otherPiece and board[i+1][j] == myPiece and board[i+2][j] == otherPiece and board[i+3][j] == otherPiece):
                return 8
            elif (i <= 2 and board[i][j] == otherPiece and board[i+1][j] == otherPiece and board[i+2][j] == myPiece and board[i+3][j] == otherPiece):
                return 8
            elif (i <= 2 and board[i][j] == otherPiece and board[i+1][j] == otherPiece and board[i+2][j] == otherPiece and board[i+3][j] == myPiece):
                return 8
            
            # blocking 3 in a row
            if (i <= 2 and board[i][j] == myPiece and board[i+1][j] == otherPiece and board[i+2][j] == otherPiece):
                return 4
            elif (i <= 2 and board[i][j] == otherPiece and board[i+1][j] == myPiece and board[i+2][j] == otherPiece):
                return 4
            elif (i <= 2 and board[i][j] == otherPiece and board[i+1][j] == otherPiece and board[i+2][j] == myPiece):
                return 4
            
            # blocking 2 in a row
            if (i <= 2 and board[i][j] == myPiece and board[i+1][j] == otherPiece):
                return 2
            elif (i <= 2 and board[i][j] == otherPiece and board[i+1][j] == myPiece):
                return 2
            

            # /////     diagonal right     /////
            # blocking 4 in a row
            if (i <= 2 and j <= 3 and board[i][j] == myPiece and board[i+1][j+1] == otherPiece and board[i+2][j+2] == otherPiece and board[i+3][j+3] == otherPiece):
                return 8
            elif (i <= 2 and j <= 3 and board[i][j] == otherPiece and board[i+1][j+1] == myPiece and board[i+2][j+2] == otherPiece and board[i+3][j+3] == otherPiece):
                return 8
            elif (i <= 2 and j <= 3 and board[i][j] == otherPiece and board[i+1][j+1] == otherPiece and board[i+2][j+2] == myPiece and board[i+3][j+3] == otherPiece):
                return 8
            elif (i <= 2 and j <= 3 and board[i][j] == otherPiece and board[i+1][j+1] == otherPiece and board[i+2][j+2] == otherPiece and board[i+3][j+3] == myPiece):
                return 8
            
            # blocking 3 in a row
            if (i <= 2 and j <= 3 and board[i][j] == myPiece and board[i+1][j+1] == otherPiece and board[i+2][j+2] == otherPiece):
                return 4
            elif (i <= 2 and j <= 3 and board[i][j] == otherPiece and board[i+1][j+1] == myPiece and board[i+2][j+2] == otherPiece):
                return 4
            elif (i <= 2 and j <= 3 and board[i][j] == otherPiece and board[i+1][j+1] == otherPiece and board[i+2][j+2] == myPiece):
                return 4
            
            # blocking 2 in a row
            if (i <= 2 and j <= 3 and board[i][j] == myPiece and board[i+1][j+1] == otherPiece):
                return 2
            elif (i <= 2 and j <= 3 and board[i][j] == otherPiece and board[i+1][j+1] == myPiece):
                return 2
            

            # /////    diagonal left   /////
            # block 4 in a row
            if ( i >= 3 and j <= 4 and board[i][j] == myPiece and board[i-1][j+1] == otherPiece and board[i-2][j+2] == otherPiece and board[i-3][j+3] == otherPiece):
                return 8
            elif ( i >= 3 and j <= 4 and board[i][j] == otherPiece and board[i-1][j+1] == myPiece and board[i-2][j+2] == otherPiece and board[i-3][j+3] == otherPiece):
                return 8
            elif ( i >= 3 and j <= 4 and board[i][j] == otherPiece and board[i-1][j+1] == otherPiece and board[i-2][j+2] == myPiece and board[i-3][j+3] == otherPiece):
                return 8
            elif ( i >= 3 and j <= 4 and board[i][j] == otherPiece and board[i-1][j+1] == otherPiece and board[i-2][j+2] == otherPiece and board[i-3][j+3] == myPiece):
                return 8
            
            # block 3 in a row
            if ( i >= 3 and j <= 4 and board[i][j] == myPiece and board[i-1][j+1] == otherPiece and board[i-2][j+2] == otherPiece):
                return 4
            # block 3 in a row
            elif ( i >= 3 and j <= 4 and board[i][j] == otherPiece and board[i-1][j+1] == myPiece and board[i-2][j+2] == otherPiece):
                return 4
            # block 3 in a row
            elif ( i >= 3 and j <= 4 and board[i][j] == otherPiece and board[i-1][j+1] == otherPiece and board[i-2][j+2] == myPiece):
                return 4
            
            # block 2 in a row
            if ( i >= 3 and j <= 4 and board[i][j] == myPiece and board[i-1][j+1] == otherPiece):
                return 2
            if ( i >= 3 and j <= 4 and board[i][j] == otherPiece and board[i-1][j+1] == myPiece):
                return 2
    
    return 0 # outcome of no block

# function to calculate punishment to currentplayer depending on the scorings of the other player
def punishWhenOpponentScores(board, piece): # potentially change name? punishing when opponent gets some kind of victory
    for i in range(6):
        for j in range(7):
            # /////     horizontal     /////
            # 4 in a row
            if (j <= 3 and board[i][j] == piece and board[i][j+1] == piece and board[i][j+2] == piece and board[i][j+3] == piece):
                return -100 # reward = -100
            # 3 in a row
            elif (j <= 3 and board[i][j] == piece and board[i][j+1] == piece and board[i][j+2] == piece):
                return -10 # reward = -10

            # /////     vertical    /////
            # 4 in a row
            if (i <= 2 and board[i][j] == piece and board[i+1][j] == piece and board[i+2][j] == piece and board[i+3][j] == piece):
                return -100
            # 3 in a row
            elif (i <= 2 and board[i][j] == piece and board[i+1][j] == piece and board[i+2][j] == piece):
                return -10

            # /////     diagonal right     /////
            # 4 in a row
            if (i <= 2 and j <= 3 and board[i][j] == piece and board[i+1][j+1] == piece and board[i+2][j+2] == piece and board[i+3][j+3] == piece):
                return -100
            # 3 in a row
            elif (i <= 2 and j <= 3 and board[i][j] == piece and board[i+1][j+1] == piece and board[i+2][j+2] == piece):
                return -10
            
            # /////    diagonal left   /////
            # 4 in a row
            if ( i >= 3 and j <= 4 and board[i][j] == piece and board[i-1][j+1] == piece and board[i-2][j+2] == piece and board[i-3][j+3] == piece):
                return -100
            # 3 in a row
            elif ( i >= 3 and j <= 4 and board[i][j] == piece and board[i-1][j+1] == piece and board[i-2][j+2] == piece):
                return -10

    return 0

def autoPlayer(board, startPlace):

    end = False

    # 1 for HORIZONTAL
    # 2 for VERTICAL 
    vertical = randint(1,3)
    direction = randint(1,2)


    # picks a random place for first piece, iterates from there
    aiRewards = 0
    turnCount = 0
    model = Connect4()

    # if direction is vertical:
    if vertical == 1:
        placeToken(board, startPlace, 1)
        return startPlace

    # if direction is horizontal:
    else:
        # if it is being placed left -> right
        if (direction == 1 or startPlace == 0) and startPlace != 6:
            placeToken(board, (startPlace + 1), 1)
            return startPlace+1


        # if it is being placed right -> left
        elif direction == 2 or startPlace == 6:
            placeToken(board, (startPlace - 1), 1)
            return startPlace-1



class C4Env(Env):
    def __init__(self):
        
        self.board = [[0]*7 for i in range(6)]
        self.startPlace = randint(1,5)
        self.action_space = spaces.Discrete(7)
        self.observation_space = self.board
        self.state = self.board

    def step(self, action):
        done = False
        reward = 1
        self.ep_return += 1

        self.startPlace = autoPlayer(self.board, self.startPlace)
        if checkWin(self.board,1) or checkWin(self.board,2) or checkFull(self.board):
            done = True

        

        if not done:
            self.board = placeToken(self.board, action-1, 2)
            if checkWin(self.board,1) or checkWin(self.board,2) or checkFull(self.board):
                done = True
        
        return self.state, reward, done

    def render(self):
        print("|1|2|3|4|5|6|7|")
        for i in range(len(self.board)):
            line = ""
            for j in range(len(self.board[i])):
                token=str(self.board[i][j])
                token = token.replace("1", "🟥")
                token = token.replace("2", "🟨")
                token = token.replace("0", "  ")
                line = line + token
            print(line)
        print("|1|2|3|4|5|6|7|")

        img = pygame.image.load("Connect4Board.png")
        img = pygame.transform.scale(img, (840, 720))
        screen.blit(img, (0, 0))
        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x] == 1:
                    img = pygame.image.load("yellow.png")
                    img = pygame.transform.scale(img, (120, 120))
                    screen.blit(img, (x*120, y*120))
                if self.board[y][x] == 2:
                    img = pygame.image.load("red.png")
                    img = pygame.transform.scale(img, (120, 120))
                    screen.blit(img, (x*120, y*120))
        pygame.display.update()
        return
            
    def reset(self):
        self.ep_return = 0
        self.board = [[0]*7 for i in range(6)]
        self.state = self.board
        return super().reset()
    


    
Env = C4Env()

episodes = 1000
for episode in range(1, episodes+1):

    for event in pygame.event.get(): 
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    state = Env.reset()
    done = False
    score = 0

    while not done:
        action = Env.action_space.sample()
        n_state, reward, done = Env.step(action)
        score+=reward
        Env.render()
    print("Episode:{},Score:{}".format(episode, score))

pygame.quit()
exit()



