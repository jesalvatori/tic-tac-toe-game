"""
Partidas usadas para avaliar as redes durante o treinamento e no teste.
Aqui ficam a contagem dos resultados e as recompensas de cada partida.
"""
import numpy as np
from t2.jogo import result
from t2.algoritmos.minimax.minimax import minimax, opponent_move

REWARDS = {'vitoria': 3., 'empate': 1., 'derrota': -3., 'invalida': -2.}


def play(network, mode, starts, seed, trace=False, measure=False):
    rng = np.random.default_rng(seed)
    board, turn = [0]*9, 1
    neural = 1 if starts else -1
    invalid, decisions, correct = 0, 0, 0
    events = []
    while result(board) is None:
        raw, outputs, delta = None, None, 0.
        if turn == neural:
            raw, move, outputs = network.choose(board, turn)
            invalid += int(board[raw] != 0)
            delta = REWARDS['invalida'] if board[raw] else 0.
            if measure:
                correct += int(raw in minimax(tuple(board), turn)[1])
            decisions += 1
        else:
            move = opponent_move(board, turn, mode, rng)
        board[move] = turn
        if trace:
            events.append(dict(board=board.copy(), actor='rede' if turn == neural else 'minimax',
                               raw=raw, move=move, outputs=outputs, reward=delta))
        turn *= -1
    end = result(board)
    outcome = 'empate' if end == 0 else ('vitoria' if end == neural else 'derrota')
    reward = REWARDS[outcome]
    if trace:
        events[-1]['reward'] += reward
    return dict(outcome=outcome, invalid=invalid, decisions=decisions, correct=correct,
                score=reward + invalid*REWARDS['invalida'], events=events)


def evaluate(network, medium_games=24, hard=True, seed=42, measure=False):
    totals = dict(vitoria=0, empate=0, derrota=0, invalid=0, decisions=0, correct=0, score=0.)
    schedule = [('medio', i%2 == 0, seed+i) for i in range(medium_games)]
    if hard:
        schedule += [('dificil', True, seed), ('dificil', False, seed)]
    for mode, starts, game_seed in schedule:
        game = play(network, mode, starts, game_seed, measure=measure)
        totals[game['outcome']] += 1
        for key in ('invalid','decisions','correct','score'):
            totals[key] += game[key]
    totals['games'] = len(schedule)
    totals['fitness'] = totals['score']/len(schedule)
    totals['valid_rate'] = 1-totals['invalid']/max(1,totals['decisions'])
    totals['accuracy'] = totals['correct']/max(1,totals['decisions']) if measure else None
    return totals

