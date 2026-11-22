"""
Rede que recebe o tabuleiro e escolhe uma casa pelas saídas calculadas.
O genético ajusta os pesos, sem backpropagation.
"""
import numpy as np

class Network:
    """
    São 9 entradas, H neurônios ocultos com tanh e 9 saídas lineares.
    Cada saída representa uma casa. Os pesos e os bias das duas camadas
    ficam juntos no cromossomo.
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
        # Lemos o tabuleiro pelo lado da rede, tanto quando ela é X quanto O.
        # A peça dela vale 1, a do adversário -1 e a casa vazia 0.
        inputs = np.asarray(board, dtype=float) * player
        activation = np.tanh(inputs @ self.w1 + self.b1)
        return activation @ self.w2 + self.b2

    def choose(self, board, player):
        outputs = self.forward(board, player)
        ranking = np.argsort(-outputs, kind='stable')
        raw = int(ranking[0])
        # Se a primeira escolha estiver ocupada, ela continua contando como erro.
        # A partida segue usando a próxima casa livre no ranking da rede.
        move = next(int(i) for i in ranking if board[i] == 0)
        return raw, move, outputs.tolist()

