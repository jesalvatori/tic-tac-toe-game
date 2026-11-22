"""
Escolha das jogadas do adversário, nos modos médio e difícil.
"""
from functools import lru_cache
from t2.apoio.jogo import result

@lru_cache(maxsize=None)
def minimax(board, player):
    """
    Calcula o valor do tabuleiro para o jogador da vez e retorna as melhores casas.
    Na hora de jogar, usamos a de menor índice quando há empate entre elas.
    """
    end = result(board)
    if end is not None:
        return end * player, ()
    scores = []
    for i, value in enumerate(board):
        if value == 0:
            child = list(board)
            child[i] = player
            score, _ = minimax(tuple(child), -player)
            scores.append((-score, i))
    best = max(s for s, _ in scores)
    return best, tuple(i for s, i in scores if s == best)


def opponent_move(board, player, mode, rng):
    available = [i for i,v in enumerate(board) if v == 0]
    if mode == 'medio' and rng.random() < .5:
        return int(rng.choice(available))
    return minimax(tuple(board), player)[1][0]

