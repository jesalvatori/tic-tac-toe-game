"""
Neste arquivo ficam as regras do jogo, a rede e o algoritmo genético.

A MLP faz a propagação direta para escolher uma posição. Quem modifica
os pesos durante o treino é o AG, sem usar backpropagation.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import numpy as np

LINES = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))

def result(board):
    for a,b,c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return 0 if all(board) else None

@lru_cache(maxsize=None)
def minimax(board, player):
    """
    Avalia o tabuleiro do ponto de vista do jogador que está na vez.

    Quando mais de uma ação tem o mesmo melhor valor, o desempate usado
    para jogar é pelo menor índice. Assim a escolha não depende de sorteio.
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

class Network:
    """
    A rede recebe 9 entradas e passa por H neurônios ocultos com tanh.

    No final temos 9 saídas lineares, uma para cada casa. Tanto a camada
    oculta quanto a de saída têm bias, que também entram no cromossomo.
    """
    def __init__(self, weights, hidden=9):
        self.hidden = hidden
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.shape != (self.size(hidden),) or not np.isfinite(self.weights).all():
            raise ValueError('Cromossomo incompatível com a topologia.')
        cut = 9 * hidden
        self.w1 = self.weights[:cut].reshape(9, hidden)
        self.b1 = self.weights[cut:cut+hidden]
        cut += hidden
        self.w2 = self.weights[cut:cut+hidden*9].reshape(hidden, 9)
        self.b2 = self.weights[-9:]

    @staticmethod
    def size(hidden):
        return 9*hidden + hidden + hidden*9 + 9

    def forward(self, board, player):
        # A entrada precisa ter o mesmo sentido quando a rede joga como X
        # ou como O. Por isso a própria peça vale 1, a do adversário vale -1
        # e a casa vazia continua valendo 0.
        inputs = np.asarray(board, dtype=float) * player
        activation = np.tanh(inputs @ self.w1 + self.b1)
        return activation @ self.w2 + self.b2

    def choose(self, board, player):
        outputs = self.forward(board, player)
        ranking = np.argsort(-outputs, kind='stable')
        raw = int(ranking[0])
        # Guardamos a primeira escolha mesmo se a casa estiver ocupada,
        # porque é esse erro que vai receber a penalidade na avaliação.
        # Para a partida não travar ali, procuramos a próxima casa livre
        # na ordem das saídas da própria rede.
        move = next(int(i) for i in ranking if board[i] == 0)
        return raw, move, outputs.tolist()

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

@dataclass
class Config:
    population: int = 80
    hidden: int = 9
    generations: int = 250
    medium_games: int = 24
    elite: int = 2
    tournament: int = 2
    mutation: float = .08
    sigma: float = .20
    alpha: float = .5
    patience: int = 60
    seed: int = 42

    def validate(self):
        ranges = {'population':(4,400), 'hidden':(1,64), 'generations':(2,3000),
                  'medium_games':(2,200), 'elite':(1,self.population-1),
                  'tournament':(2,self.population), 'patience':(2,3000), 'seed':(0,2**31-1)}
        for k,(low,high) in ranges.items():
            v = getattr(self,k)
            if type(v) is not int or not low <= v <= high:
                raise ValueError(f'{k}: use inteiro entre {low} e {high}.')
        for k,low,high in [('mutation',0,1),('sigma',.001,1),('alpha',0,1)]:
            v = getattr(self,k)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not low <= v <= high:
                raise ValueError(f'{k}: valor fora do intervalo {low} a {high}.')
        if self.medium_games % 2:
            raise ValueError('Use número par de partidas para equilibrar quem começa.')

def crossover(a, b, rng, alpha):
    # No BLX-alpha, cada gene recebe seu próprio beta, sorteado no intervalo
    # [-alpha, 1+alpha]. A combinação pode gerar um peso diferente dos dois
    # pais, em vez de só copiar um dos valores que já existiam.
    beta = rng.uniform(-alpha,1+alpha,len(a))
    return np.clip(a + beta*(b-a), -1, 1)

class Trainer:
    def __init__(self, config):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.population = self.rng.uniform(-1,1,(config.population, Network.size(config.hidden)))
        self.generation = 0
        self.history, self.ranked = [], []
        self.best = None
        self.best_fitness = float('-inf')
        self.stagnant, self.previous_phase = 0, None
        self.stopped = None

    def step(self):
        if self.stopped:
            return self.state()
        c = self.config
        hard = self.generation >= max(1,int(c.generations*.4))
        phase = 'medio+dificil' if hard else 'medio'
        # Quando entra o modo difícil, o conjunto de partidas da avaliação
        # muda. Reiniciamos a comparação para não tratar a aptidão da fase
        # anterior como se ela tivesse sido obtida nas mesmas condições.
        if phase != self.previous_phase:
            self.best_fitness, self.stagnant = float('-inf'), 0
            self.previous_phase = phase
        if self.generation:
            parents = self.population
            next_population = [parents[i].copy() for i in range(c.elite)]
            def tournament():
                # A população já está ordenada da maior aptidão para a menor.
                # Então o menor índice entre os sorteados é justamente o
                # melhor candidato desse torneio.
                return parents[int(self.rng.choice(c.population,c.tournament,replace=False).min())]
            while len(next_population) < c.population:
                child = crossover(tournament(),tournament(),self.rng,c.alpha)
                mask = self.rng.random(len(child)) < c.mutation
                child[mask] += self.rng.normal(0,c.sigma,int(mask.sum()))
                next_population.append(np.clip(child,-1,1))
            self.population = np.array(next_population)
        # Todos os indivíduos são avaliados com as mesmas sementes. Assim,
        # uma rede não ganha vantagem só por receber outros sorteios no modo
        # médio, e conseguimos repetir a avaliação com a mesma configuração.
        results = [evaluate(Network(w,c.hidden),c.medium_games,hard,c.seed) for w in self.population]
        order = sorted(range(c.population),key=lambda i:results[i]['fitness'],reverse=True)
        self.population = self.population[order]
        self.ranked = [results[i] for i in order]
        top = self.ranked[0]
        self.best = Network(self.population[0].copy(),c.hidden)
        if top['fitness'] > self.best_fitness + 1e-6:
            self.best_fitness, self.stagnant = top['fitness'], 0
        else:
            self.stagnant += 1
        self.generation += 1
        self.history.append(dict(generation=self.generation, phase=phase,
                                 best=top['fitness'], mean=float(np.mean([r['fitness'] for r in results])),
                                 valid_rate=top['valid_rate'], vitoria=top['vitoria'],
                                 empate=top['empate'], derrota=top['derrota'], invalid=top['invalid']))
        if self.generation >= c.generations:
            self.stopped = 'Limite de gerações'
        elif hard and self.stagnant >= c.patience:
            self.stopped = 'Convergência: aptidão estável no conjunto de treino'
        return self.state()

    def state(self):
        return dict(config=asdict(self.config), generation=self.generation, history=self.history,
                    population=self.ranked, stopped=self.stopped,
                    parameters=Network.size(self.config.hidden), rewards=REWARDS)
