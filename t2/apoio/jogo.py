"""
Regras do tabuleiro e verificação de vitória ou empate.
"""


LINES = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))


def result(board):
    for a,b,c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return 0 if all(board) else None

