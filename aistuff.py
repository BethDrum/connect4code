import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.spaces import discrete
from random import random, randint
from gymnasium import Env, spaces
import pygame
from collections import deque

pygame.init()
SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Connect4")
pygame.display.update()
font = pygame.font.SysFont('Corbel',35)
color = (255,255,255) 


class Connect4(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6*7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.ReLU(),
            nn.Linear(250, 7),
            #nn.Sigmoid()
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
        if board [0][i] == 0:
            return False
    return True    

def placeToken(board, j, piece):
    for i in reversed(range(6)):
        if board[i][j] == 0:
            board[i][j] = piece
            return board
    return board

#               PLEASE CALL THESE FUNCTIONS /BEFORE/ CHECKWIN. IF CHECKWIN IS CALLED BEFORE, NO REWARDS ARE AWARDED FOR WINNING

# function calculate rewards gathered from 4's, 3's, 2's, 1's in a row
def rewardRows(board, piece):
    reward = 1
    for i in range(6):
        for j in range(7):

            # --------- HORIZONTAL ---------
            if j <= 3:
                # 4 in a row
                if (board[i][j] == piece and board[i][j+1] == piece and board[i][j+2] == piece and board[i][j+3] == piece):
                    reward += 100000000 
                
                # 3 in a row 
                elif (board[i][j] == piece and board[i][j+1] == piece and board[i][j+2] == piece):
                    reward += 80
                
                # 2 in a row 
                elif (board[i][j] == piece and board[i][j+1] == piece):
                    reward += 2

            # /////     Vertical     /////
            if i <= 2:
                # 4 in a row
                if (board[i][j] == piece and board[i+1][j] == piece and board[i+2][j] == piece and board[i+3][j] == piece):
                    reward += 100000000 
                
                # 3 in a row
                elif (board[i][j] == piece and board[i+1][j] == piece and board[i+2][j] == piece):
                    reward += 80
                
                # 2 in a row
                elif (board[i][j] == piece and board[i+1][j] == piece):
                    reward += 2

            # /////     diagonal right     /////
            if i <= 2 and j <= 3:
                # 4 in a row
                if (board[i][j] == piece and board[i+1][j+1] == piece and board[i+2][j+2] == piece and board[i+3][j+3] == piece):
                    reward += 100000000 
                # 3 in a row
                elif (board[i][j] == piece and board[i+1][j+1] == piece and board[i+2][j+2] == piece):
                    reward += 80
                # 2 in a row
                elif (board[i][j] == piece and 
                    board[i+1][j+1] == piece):
                    reward += 2

            # /////     diagonal left     /////
            if i >= 3 and j <= 3:
                # 4 in a row
                if (board[i][j] == piece and board[i-1][j+1] == piece and board[i-2][j+2] == piece and board[i-3][j+3] == piece):
                    reward += 100000000 
                
                # 3 in a row
                elif (board[i][j] == piece and board[i-1][j+1] == piece and board[i-2][j+2] == piece):
                    reward += 80
                
                # 2 in a row
                elif (board[i][j] == piece and board[i-1][j+1] == piece):
                    reward += 2

    return reward

# function to calculate rewards gathered from blocking opponent plays
def rewardOpponentBlock(board, myPiece, otherPiece):
    reward = 0
    for i in range (6):
        for j in range (7):
            # /////     horizontal     /////
            # blocking a 3 in a row
            if (j <= 3 and board[i][j] == myPiece and board[i][j+1] == otherPiece and board[i][j+2] == otherPiece and board[i][j+3] == otherPiece):
                reward += 8 # reward = 8
            elif (j <= 3 and board[i][j] == otherPiece and board[i][j+1] == myPiece and board[i][j+2] == otherPiece and board[i][j+3] == otherPiece):
                reward += 8 
            elif (j <= 3 and board[i][j] == otherPiece and board[i][j+1] == otherPiece and board[i][j+2] == myPiece and board[i][j+3] == otherPiece):
                reward += 8    
            elif (j <= 3 and board[i][j] == otherPiece and board[i][j+1] == otherPiece and board[i][j+2] == otherPiece and board[i][j+3] == myPiece):
                reward += 8 

            # blocking a 2 in a row
            if (j <= 3 and board[i][j] == myPiece and board[i][j+1] == otherPiece and board[i][j+2] == otherPiece):
                reward += 4 # reward = 4
            elif (j <= 3 and board[i][j] == otherPiece and board[i][j+1] == myPiece and board[i][j+2] == otherPiece):
                reward += 4 
            elif (j <= 3 and board[i][j] == otherPiece and board[i][j+1] == otherPiece and board[i][j+2] == myPiece):
                reward += 4 

            # blocking a 1 in a row
            if (j <= 3 and board[i][j] == myPiece and board[i][j+1] == otherPiece):
                reward += 2 # reward # 2
            elif (j <= 3 and board[i][j] == otherPiece and board[i][j+1] == myPiece):
                reward += 2 # reward # 2


            # /////     vertical    /////
            # blocking 4 in a row
            if (i <= 2 and board[i][j] == myPiece and board[i+1][j] == otherPiece and board[i+2][j] == otherPiece and board[i+3][j] == otherPiece):
                reward += 8
            elif (i <= 2 and board[i][j] == otherPiece and board[i+1][j] == myPiece and board[i+2][j] == otherPiece and board[i+3][j] == otherPiece):
                reward += 8
            elif (i <= 2 and board[i][j] == otherPiece and board[i+1][j] == otherPiece and board[i+2][j] == myPiece and board[i+3][j] == otherPiece):
                reward += 8
            elif (i <= 2 and board[i][j] == otherPiece and board[i+1][j] == otherPiece and board[i+2][j] == otherPiece and board[i+3][j] == myPiece):
                reward += 8
            
            # blocking 3 in a row
            if (i <= 2 and board[i][j] == myPiece and board[i+1][j] == otherPiece and board[i+2][j] == otherPiece):
                reward += 4
            elif (i <= 2 and board[i][j] == otherPiece and board[i+1][j] == myPiece and board[i+2][j] == otherPiece):
                reward += 4
            elif (i <= 2 and board[i][j] == otherPiece and board[i+1][j] == otherPiece and board[i+2][j] == myPiece):
                reward += 4
            
            # blocking 2 in a row
            if (i <= 2 and board[i][j] == myPiece and board[i+1][j] == otherPiece):
                reward += 2
            elif (i <= 2 and board[i][j] == otherPiece and board[i+1][j] == myPiece):
                reward += 2
            

            # /////     diagonal right     /////
            # blocking 4 in a row
            if (i <= 2 and j <= 3 and board[i][j] == myPiece and board[i+1][j+1] == otherPiece and board[i+2][j+2] == otherPiece and board[i+3][j+3] == otherPiece):
                reward += 8
            elif (i <= 2 and j <= 3 and board[i][j] == otherPiece and board[i+1][j+1] == myPiece and board[i+2][j+2] == otherPiece and board[i+3][j+3] == otherPiece):
                reward += 8
            elif (i <= 2 and j <= 3 and board[i][j] == otherPiece and board[i+1][j+1] == otherPiece and board[i+2][j+2] == myPiece and board[i+3][j+3] == otherPiece):
                reward += 8
            elif (i <= 2 and j <= 3 and board[i][j] == otherPiece and board[i+1][j+1] == otherPiece and board[i+2][j+2] == otherPiece and board[i+3][j+3] == myPiece):
                reward += 8
            
            # blocking 3 in a row
            if (i <= 2 and j <= 3 and board[i][j] == myPiece and board[i+1][j+1] == otherPiece and board[i+2][j+2] == otherPiece):
                reward += 4
            elif (i <= 2 and j <= 3 and board[i][j] == otherPiece and board[i+1][j+1] == myPiece and board[i+2][j+2] == otherPiece):
                reward += 4
            elif (i <= 2 and j <= 3 and board[i][j] == otherPiece and board[i+1][j+1] == otherPiece and board[i+2][j+2] == myPiece):
                reward += 4
            
            # blocking 2 in a row
            if (i <= 2 and j <= 3 and board[i][j] == myPiece and board[i+1][j+1] == otherPiece):
                reward += 2
            elif (i <= 2 and j <= 3 and board[i][j] == otherPiece and board[i+1][j+1] == myPiece):
                reward += 2
            

            # /////    diagonal left   /////
            # block 4 in a row
            if ( i >= 3 and j <= 3 and board[i][j] == myPiece and board[i-1][j+1] == otherPiece and board[i-2][j+2] == otherPiece and board[i-3][j+3] == otherPiece):
                reward += 8
            elif ( i >= 3 and j <= 3 and board[i][j] == otherPiece and board[i-1][j+1] == myPiece and board[i-2][j+2] == otherPiece and board[i-3][j+3] == otherPiece):
                reward += 8
            elif ( i >= 3 and j <= 3 and board[i][j] == otherPiece and board[i-1][j+1] == otherPiece and board[i-2][j+2] == myPiece and board[i-3][j+3] == otherPiece):
                reward += 8
            elif ( i >= 3 and j <= 3 and board[i][j] == otherPiece and board[i-1][j+1] == otherPiece and board[i-2][j+2] == otherPiece and board[i-3][j+3] == myPiece):
                reward += 8
            
            # block 3 in a row
            if ( i >= 3 and j <= 3 and board[i][j] == myPiece and board[i-1][j+1] == otherPiece and board[i-2][j+2] == otherPiece):
                reward += 4
            # block 3 in a row
            elif ( i >= 3 and j <= 3 and board[i][j] == otherPiece and board[i-1][j+1] == myPiece and board[i-2][j+2] == otherPiece):
                reward += 4
            # block 3 in a row
            elif ( i >= 3 and j <= 3 and board[i][j] == otherPiece and board[i-1][j+1] == otherPiece and board[i-2][j+2] == myPiece):
                reward += 4
            
            # block 2 in a row
            if ( i >= 3 and j <= 3 and board[i][j] == myPiece and board[i-1][j+1] == otherPiece):
                reward += 2
            if ( i >= 3 and j <= 3 and board[i][j] == otherPiece and board[i-1][j+1] == myPiece):
                reward += 2
    
    return reward # outcome of no block

# function to calculate punishment to currentplayer depending on the scorings of the other player
def punishWhenOpponentScores(board, piece): # potentially change name? punishing when opponent gets some kind of victory
    reward = 0
    for i in range(6):
        for j in range(7):
            
            # /////     horizontal     /////
            # 4 in a row
            if (j <= 3 and board[i][j] == piece and board[i][j+1] == piece and board[i][j+2] == piece and board[i][j+3] == piece):
                reward += -100000000 # reward = -100
                
            # 3 in a row
            elif (j <= 3 and board[i][j] == piece and board[i][j+1] == piece and board[i][j+2] == piece):
                reward += -10 # reward = -10

            # /////     vertical    /////
            # 4 in a row
            if (i <= 2 and board[i][j] == piece and board[i+1][j] == piece and board[i+2][j] == piece and board[i+3][j] == piece):
                reward += -100000000
                
            # 3 in a row
            elif (i <= 2 and board[i][j] == piece and board[i+1][j] == piece and board[i+2][j] == piece):
                reward += -10

            # /////     diagonal right     /////
            # 4 in a row
            if (i <= 2 and j <= 3 and board[i][j] == piece and board[i+1][j+1] == piece and board[i+2][j+2] == piece and board[i+3][j+3] == piece):
                reward += -100000000
                
            # 3 in a row
            elif (i <= 2 and j <= 3 and board[i][j] == piece and board[i+1][j+1] == piece and board[i+2][j+2] == piece):
                reward += -10
            
            # /////    diagonal left   /////
            # 4 in a row
            if ( i >= 3 and j <= 3 and board[i][j] == piece and board[i-1][j+1] == piece and board[i-2][j+2] == piece and board[i-3][j+3] == piece):
                reward += -100000000
                
            # 3 in a row
            elif ( i >= 3 and j <= 3 and board[i][j] == piece and board[i-1][j+1] == piece and board[i-2][j+2] == piece):
                reward += -10

    return reward

def autoPlayer(board, startPlace):

    end = False

    # 1 for HORIZONTAL
    # 2 for VERTICAL 
    vertical = randint(1,3)
    direction = randint(1,2)


    # picks a random place for first piece, iterates from there
    aiRewards = 0
    turnCount = 0

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

class ReplayMemory(object):
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def add_experience(self, state, action, reward, next_state):
        experience = (state, action, reward, next_state)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        sample_batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*sample_batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)
    
    def size(self):
        return len(self.buffer)

class C4Env(Env):
    def __init__(self):
        
        self.board = [[0]*7 for i in range(6)]
        self.startPlace = randint(1,5)
        self.action_space = spaces.Discrete(7)
        self.observation_space = self.board
        self.state = self.board
        self.model = Connect4()
        self.turnCount = 0
        self.loss_fn = nn.HuberLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.replayBuffer = ReplayMemory(10000)
        self.avg = 0

    def step(self,action, player):
        done = False
        reward = -1
        self.ep_return += 1
        aiRewards = 0

        if not player:
            self.startPlace = autoPlayer(self.board, self.startPlace)
            if checkWin(self.board,1) or checkWin(self.board,2) or checkFull(self.board):
                done = True
        else:
            col = waitForClick()
            board = placeToken(self.board, col-1, 1)

        y = torch.tensor(self.board, dtype=torch.float32).reshape(1, 42)
        t = self.model(y)
        expected_reward = t.argmax(1).item()
        predic = nn.Softmax(dim=1)(t)
        #pick the highest value and play that piece
        yPred = predic.argmax(1).item()
        self.turnCount += 1
        
        if not done:
            self.board = placeToken(self.board, action-1, 2)
            if checkWin(self.board,1) or checkFull(self.board):
                done = True
                
            
            

            if checkWin(self.board,2):
                self.avg += 1
                done = True

        if player:
            self.render(0,0)

        aiRewards += rewardRows(self.board, 2)
        aiRewards += rewardOpponentBlock(self.board, 2,1)
        aiRewards += punishWhenOpponentScores(self.board, 1)
        #print(aiRewards)
        #print("reward")
        #reward = (aiRewards*self.turnCount - aiRewards)
        reward = aiRewards
        self.optimizer.zero_grad()
        self.loss_fn(predic, predic*aiRewards).backward()
        self.optimizer.step()
        
        return self.state, reward, done, self.avg

    def render(self, episode, score):
        screen.fill((0,0,0))
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

        if (episode != 0):
            episode = font.render('Episode:' + str(episode), True , color)
            screen.blit(episode , (860,10))
        score = font.render('Score:' + str(score), True , color)
        screen.blit(score , (860,50))

        pygame.display.update()
        return
            
    def reset(self):
        self.ep_return = 0
        self.board = [[0]*7 for i in range(6)]
        self.state = self.board
        return super().reset()
    

def waitForClick():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                COLUMN_WIDTH = 120
                col = mouse_x // COLUMN_WIDTH   # 0–6
                return col + 1

def displayBoard(board):

    print("|1|2|3|4|5|6|7|")
    for i in range(len(board)):
        line = ""
        for j in range(len(board[i])):
            token=str(board[i][j])
            token = token.replace("1", "🟥")
            token = token.replace("2", "🟨")
            token = token.replace("0", "  ")
            line = line + token
        print(line)
    print("|1|2|3|4|5|6|7|")

    img = pygame.image.load("Connect4Board.png")
    img = pygame.transform.scale(img, (840, 720))
    screen.blit(img, (0, 0))
    for y in range(len(board)):
        for x in range(len(board[y])):
            if board[y][x] == 1:
                img = pygame.image.load("yellow.png")
                img = pygame.transform.scale(img, (120, 120))
                screen.blit(img, (x*120, y*120))
            if board[y][x] == 2:
                img = pygame.image.load("red.png")
                img = pygame.transform.scale(img, (120, 120))
                screen.blit(img, (x*120, y*120))
    pygame.display.update()
    
Env = C4Env()

episodes = 500000
avgwin = 0
for episode in range(1, episodes+1):
    player = False
    
    if episode > 10000:
        print(avgwin/10000)
        player = True


    for event in pygame.event.get(): 
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    state = Env.reset()
    done = False
    score = 0

    while not done:
        action = Env.action_space.sample()
        n_state, reward, done, avgwin = Env.step(action, player)
        score=reward
        #makes it better to display ever so often
    if episode%50 == 0:
        Env.render(episode, score)
        print("Episode:{},Score:{}".format(episode, score))

    

run = True
board = [[0]*7 for i in range (6)]
while run == True:
    displayBoard(board)
    for event in pygame.event.get(): 
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
    print("Player 1, please enter the column you wish to add your piece in: \n")
    col = waitForClick()
    board = placeToken(board, col-1, 1)
    win = checkWin(board, 1)
    displayBoard(board)
    if win == 'won':
        print("Player 1 wins!")
        run = False
    #turn the board (6x7) to a tensor (matrix of matrices) that is 1x42 (so basically making our
    # board into one LONG line of numbers by just making it have no rows lol)
    y = torch.tensor(board, dtype=torch.float32).reshape(1, 42)
    #make a copy of the model's result, and make your prediction
    t = Env.model(y)
    predic = nn.Softmax(dim=1)(t)
    #pick the highest value and play that piece
    yPred = predic.argmax(1).item()
    placeToken(board, yPred, 2)
    displayBoard(board)
    win = checkWin(board,2)
    if win == True:
        run = False
        print("AI won")
    # calculate rewards gathered from the previous play
    # WHEN PLACING 2 AI AGAINST EACHOTHER, THESE VALUES WILL BE REVERSED. EACH AI WOULD NEED ITS OWN REWARD VARIABLE
    #show each prediction value just for testing sake
    print(predic)
    print("ai %i" %yPred)
pygame.quit()
exit()



