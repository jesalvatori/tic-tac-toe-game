"""
O algoritmo genético monta a população e evolui os pesos das redes.
A avaliação das partidas fica no arquivo de treinamento.
"""
from dataclasses import dataclass, asdict
import numpy as np
from t2.algoritmos.mlp.mlp import Network
from t2.apoio.treinamento import REWARDS, evaluate

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

