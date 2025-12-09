import aistuff
import pygame
import random;

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
    COLUMN_WIDTH = 120   # = 120 pixels per column
    col = mouse_x / COLUMN_WIDTH   # 0–6
    return col + 1  

def waitForClick():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                COLUMN_WIDTH = 120
                col = mouse_x / COLUMN_WIDTH   # 0–6
                return col + 1

def autoPlayer(board):

    end = False

    # 1 for HORIZONTAL
    # 2 for VERTICAL 
    direction = randint(1,2)

    # picks a random place for first piece, iterates from there
    startPlace = randint(0,6)

    # if direction is horizontal:
    if direction == 1:
        # if it is being placed left -> right
        if startPlace <= 4:
            for i in range(4):
                placeToken(board, (startPlace + i), 1)

                # check if auto-player has won
                win = checkWin(board, 1)
                if win == "won":
                    print("Player 1 wins!")
                    end = True;

        # if it is being placed right -> left
        elif startPlace > 4:
            for i in range(4):
                placeToken(board, (startPlace - i), 1)

                # check if auto-player has won
                win = checkWin(board, 1)
                if win == "won":
                    print("Player 1 wins!")   
                    end = True;

    # if direction is vertical:
    elif direction == 2:
        for i in range(4):
            placeToken(board, startPlace, 1)
            
            # check if auto-player has won
            win = checkWin(board, 1)
            if win == "won":
                print("Player 1 wins!")
                end = True;


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
    #changes board to be 0s, 1s and 2s (sorry bethany, this just helped my head and the ai's empty dumbass brain)
    
    model = aistuff.Connect4()  #makes the model object basically
    while True:
        displayBoard(board)
        print("Player 1, please enter the column you wish to add your piece in: \n")
        col = waitForClick()
        board = placeToken(board, col-1, 1)
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
        displayBoard(board)
        #show each prediction value just for testing sake
        print(predic)
        print("ai %i" %yPred)

def checkFull(board):
    for i in range (7):
        if board [0][i] != 0:
            return True
    return False

#main code
board = Initalise()
play2players(board)

