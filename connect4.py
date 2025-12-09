import aistuff
import pygame

#initalise pygame and screen
pygame.init()
SCREEN_WIDTH = 840
SCREEN_HEIGHT = 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Connect4")
pygame.display.update()

def Initalise ():
    board = [[0]*7 for i in range(6)]
    return board

#place a token into the board, using a reversed range loop
def placeToken(board, j, piece):
    for i in reversed(range(6)):
        if board[i][j] == 0:
            board[i][j] = piece
            return board
    return board

#check if a player has won or not
def checkWin(board, piece):
    for i in range(6):
        for j in range(7):
            if (j <= 3 and board[i][j] == piece and board[i][j+1] == piece and board[i][j+2] == piece and board[i][j+3] == piece):
                return 'won'
            #vertical
            if (i <= 2 and board[i][j] == piece and board[i+1][j] == piece and board[i+2][j] == piece and board[i+3][j] == piece):
                return 'won'
            #diagonal right 
            if (i <= 2 and j <= 3 and board[i][j] == piece and board[i+1][j+1] == piece and board[i+2][j+2] == piece and board[i+3][j+3] == piece):
                return 'won'
            #diagonal left 
            if ( i >= 3 and j <= 4 and board[i][j] == piece and board[i-1][j+1] == piece and board[i-2][j+2] == piece and board[i-3][j+3] == piece):
                return 'won'
    return 'noWin'

#                                             ////////// REWARD FUNCTIONALITY //////////
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

def getColumnFromClick(mouse_x):
    COLUMN_WIDTH = 840 // 7   # = 102 pixels per column
    col = mouse_x // COLUMN_WIDTH   # 0–6
    return col + 1  

def waitForClick():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                COLUMN_WIDTH = 840 // 7
                col = mouse_x // COLUMN_WIDTH   # 0–6
                return col + 1

def play2players(board):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        displayBoard(board)
        print("Player 1, please enter the column you wish to add your piece in: \n")
        col = waitForClick()
        board = placeToken(board, col-1, 1)
        win = checkWin(board, 1)
        if win == 'won':
            print("Player 1 wins!")
            return True
        
        displayBoard(board)
        print("Player 2, please enter the column you wish to add your piece in: \n")
        col = waitForClick()
        board = placeToken(board, col-1, 2)
        win = checkWin(board, 2)
        if win == 'won':
            print("Player 1 wins!")
            return False

# ripped most of the 2 player method here
def playAi(board):
    # variable to store all rewards the ai gains during gameplay
    aiRewards = 0

    # create model object 
    model = aistuff.Connect4()

    # running the game
    while True:
        displayBoard(board)
        print("Player 1, please enter the column you wish to add your piece in: \n")
        col = waitForClick()
        board = placeToken(board, col-1, 1)

        # check for win status
        win = checkWin(board, 1)
        if win == 'won':
            print("Player 1 wins!")
            return True
        
        #turn the board (6x7) to a tensor (matrix of matrices) that is 1x42 (so basically making our
        # board into one LONG line of numbers by just making it have no rows lol)
        y = aistuff.torch.tensor(board, dtype=aistuff.torch.float32).reshape(1, 42)

        #make a copy of the model's result, and make your prediction
        t = model(y)
        predic = aistuff.nn.Softmax(dim=1)(t)

        #pick the highest value and play that piece
        yPred = predic.argmax(1).item()
        placeToken(board, yPred, 2)

        # calculate rewards gathered from the previous play
        # WHEN PLACING 2 AI AGAINST EACHOTHER, THESE VALUES WILL BE REVERSED. EACH AI WOULD NEED ITS OWN REWARD VARIABLE
        aiRewards = aiRewards+rewardRows(board, 2)
        aiRewards = aiRewards+rewardOpponentBlock(board, 2,1)
        aiRewards = aiRewards+punishWhenOpponentScores(board, 1)
        print(aiRewards)

        displayBoard(board)

        # check for win status
        win = checkWin(board, 2)
        if win == 'won':
            print("Player 2 wins!")
            return True
        
        #show each prediction value just for testing sake
        print(predic)
        print("ai %i" %yPred)

#main code
board = Initalise()
playAi(board)