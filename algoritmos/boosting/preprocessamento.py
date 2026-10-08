"""
T1 - Tic Tac Toe com ML | Etapa 3: pré-processamento

Abordagem 1: o próprio tabuleiro, convertido em números
Abordagem 2: features derivadas pedidas
"""
import numpy as np
import pandas as pd

COLUNAS = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br"]
LINHAS = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
          (0, 3, 6), (1, 4, 7), (2, 5, 8),
          (0, 4, 8), (2, 4, 6)]


MAPA = {"x": 1, "o": -1, "b": 0}


def abordagem1(df):
    """Tabuleiro atual em valores numéricos."""
    return df[COLUNAS].replace(MAPA).astype(int)


def abordagem2(df):
    linhas = []
    for tab in df[COLUNAS].values:
        nx = int((tab == "x").sum())
        no = int((tab == "o").sum())
        ocupadas = [int(c != "b") for c in tab]
        linhas_2x = sum(1 for l in LINHAS if sum(tab[i] == "x" for i in l) == 2)
        linhas_2o = sum(1 for l in LINHAS if sum(tab[i] == "o" for i in l) == 2)
        vez = 1 if nx == no else 0  # 1 = vez do X, 0 = vez do O
        linhas.append([nx, no, *ocupadas, linhas_2x, linhas_2o, 9 - nx - no, vez])

    nomes = (["qtd_x", "qtd_o"] + [f"ocup_{c}" for c in COLUNAS]
             + ["linhas_2x", "linhas_2o", "casas_vazias", "vez_x"])
    return pd.DataFrame(linhas, columns=nomes, index=df.index)


def carregar(abordagem):
    f = abordagem1 if abordagem == 1 else abordagem2
    saida = []
    for nome in ["treino", "validacao", "teste"]:
        df = pd.read_csv(f"{nome}.csv")
        saida += [f(df), df["classe"]]
    return saida
