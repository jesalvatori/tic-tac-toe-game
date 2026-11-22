from pathlib import Path

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

RAIZ = Path(__file__).resolve().parents[2]
VALORES_K = [1, 3, 5, 7, 9, 11, 15, 21]

CLASSES = [
    "Tem jogo",
    "X venceu",
    "O venceu",
    "Empate",
]


def carregar_dados(abordagem):
    pasta = RAIZ / "dataset" / abordagem

    treino = pd.read_csv(pasta / "treino.csv")
    validacao = pd.read_csv(pasta / "validacao.csv")
    teste = pd.read_csv(pasta / "teste.csv")

    return treino, validacao, teste


def separar_dados(dados):
    X = dados.drop(columns=["classe"])
    y = dados["classe"]
    return X, y


def criar_modelo(k):
    return Pipeline([
        ("normalizacao", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])


def executar():
    melhor_f1 = -1
    melhor_k = None
    melhor_abordagem = None
    melhor_modelo = None

    for abordagem in ["abordagem_1", "abordagem_2"]:
        treino, validacao, _ = carregar_dados(abordagem)

        X_treino, y_treino = separar_dados(treino)
        X_validacao, y_validacao = separar_dados(validacao)

        print(f"\n{abordagem.upper()}")

        for k in VALORES_K:
            modelo = criar_modelo(k)
            modelo.fit(X_treino, y_treino)

            previsoes = modelo.predict(X_validacao)

            f1 = f1_score(
                y_validacao,
                previsoes,
                average="macro",
                zero_division=0,
            )

            print(f"K = {k} | F1 macro = {f1:.4f}")

            if f1 > melhor_f1:
                melhor_f1 = f1
                melhor_k = k
                melhor_abordagem = abordagem
                melhor_modelo = modelo

    print("\nMELHOR CONFIGURACAO DO KNN")
    print(f"Abordagem: {melhor_abordagem}")
    print(f"K: {melhor_k}")
    print(f"F1 macro na validacao: {melhor_f1:.4f}")

    _, _, teste = carregar_dados(melhor_abordagem)
    X_teste, y_teste = separar_dados(teste)

    previsoes_teste = melhor_modelo.predict(X_teste)

    acuracia = accuracy_score(y_teste, previsoes_teste)
    precisao = precision_score(
        y_teste, previsoes_teste,
        average="macro", zero_division=0,
    )
    recall = recall_score(
        y_teste, previsoes_teste,
        average="macro", zero_division=0,
    )
    f1_teste = f1_score(
        y_teste, previsoes_teste,
        average="macro", zero_division=0,
    )

    print("\nRESULTADOS NO TESTE")
    print(f"Acuracia: {acuracia:.4f}")
    print(f"Precisao macro: {precisao:.4f}")
    print(f"Recall macro: {recall:.4f}")
    print(f"F1 macro: {f1_teste:.4f}")

    print("\nRELATORIO POR CLASSE")
    print(classification_report(
        y_teste,
        previsoes_teste,
        labels=CLASSES,
        zero_division=0,
    ))

    print("\nMATRIZ DE CONFUSAO")
    print("Ordem das classes:", CLASSES)
    print(confusion_matrix(
        y_teste,
        previsoes_teste,
        labels=CLASSES,
    ))


if __name__ == "__main__":
    executar()