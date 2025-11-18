def putValInBoard (board):
    board = [["|", "|", "|", "|", "|", "|", "|"],
             ["|", "|", "|", "|", "|", "|", "|"], 
             ["|", "|", "|", "|", "|", "|", "|"], 
             ["|", "|", "|", "|", "|", "|", "|"], 
             ["|", "|", "|", "|", "|", "|", "|"], 
             ["|", "|", "|", "|", "|", "|", "|"]]
    return board

#place a token into the board, using a reversed range loop
def placeToken(board, j, piece):
    for i in reversed(range(6)):
        if board[i][j] == '|':
            board[i][j] = piece
            return board
    return board

#check if a player has won or not

def checkWin(board, piece):
    for i in range(0, 5):
        for j in range(0, 6):
            if (j <= 2 and board[i][j] == piece and board[i][j+1] == piece and board[i][j+2] == piece and board[i][j+3] == piece):
                return 'won'
            #vertical
            if (i <= 3 and board[i][j] == piece and board[i+1][j] == piece and board[i+2][j] == piece and board[i+3][j] == piece):
                return 'won'
            #diagonal right 
            if (i <= 3 and j <= 4 and board[i][j] == piece and board[i+1][j+1] == piece and board[i+2][j+2] == piece and board[i+3][j+3] == piece):
                return 'won'
            #diagonal left 
            if ( i >= 3 and j <= 4 and board[i][j] == piece and board[i-1][j+1] == piece and board[i-2][j+2] == piece and board[i-3][j+3] == piece):
                return 'won'
    return 'noWin'

def displayBoard(board):
    for i in board:
        print(i)
    print()

def play2players(board):
    while True:
        displayBoard(board)
        print("Player 1, please enter the column you wish to add your piece in: \n")
        col = int(input())
        board = placeToken(board, col, 'x')
        win = checkWin(board, 'x')
        if win == 'won':
            print("Player 1 wins!")
            return True
        
        displayBoard(board)
        print("Player 2, please enter the column you wish to add your piece in: \n")
        int(col = input())
        board = placeToken(board, col, 'o')
        win = checkWin(board, 'o')
        if win == 'won':
            print("Player 1 wins!")
            return False



#main code
board = [['|']*7 for i in range(6)]
board = putValInBoard(board)
play2players(board)

#for i in board:
#    print(i)
#print()
#placeToken(board, 2, 'x')

#board = putValInBoard(board)
#displayBoard(board)
#win = checkWin(board, 'x')
#if win == "won":
#    print("win")
#else:
#    print("No win")