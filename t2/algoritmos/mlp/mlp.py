"""
A rede recebe o tabuleiro e calcula as saídas para escolher uma casa.
Os pesos chegam do algoritmo genético, sem usar backpropagation.
"""
import numpy as np

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

