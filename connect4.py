import aistuff

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
    for i in range(0, 5):
        for j in range(0, 6):
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

def play2players(board):
    while True:
        displayBoard(board)
        print("Player 1, please enter the column you wish to add your piece in: \n")
        col = int(input())
        board = placeToken(board, col-1, 1)
        win = checkWin(board, 1)
        if win == 'won':
            print("Player 1 wins!")
            return True
        
        displayBoard(board)
        print("Player 2, please enter the column you wish to add your piece in: \n")
        col = int(input())
        board = placeToken(board, col-1, 2)
        win = checkWin(board, 2)
        if win == 'won':
            print("Player 1 wins!")
            return False

# ripped most of the 2 player method here
def playAi(board):
    #changes board to be 0s, 1s and 2s (sorry bethany, this just helped my head and the ai's empty dumbass brain)
    listFix(board) 
    
    model = aistuff.Connect4()  #makes the model object basically
    while True:
        displayBoard(board)
        print("Player 1, please enter the column you wish to add your piece in: \n")
        col = int(input())
        board = placeToken(board, col, 1) 
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

#method to change the board to be numbers 
def listFix(board):
    for i in range (6):
        for j in range (7):
            if(board[i][j] == '|'):
                board[i][j] = 0
            else:
                board[i][j] = 1 if board[i][j] == 'x' else 2
    return board



#main code
board = Initalise()
play2players(board)

