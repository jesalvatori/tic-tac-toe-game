# Árvore de Decisão - T1 Tic Tac Toe
import itertools
import time

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.tree import DecisionTreeClassifier

from preprocessamento import carregar

SEMENTE = 42

# Parâmetros testados
GRADE = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
    "min_samples_leaf": [1, 2, 5, 10],
    "class_weight": [None, "balanced"],
}


def metricas(y_real, y_pred):
    return {
        "acuracia": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_real, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_real, y_pred, average="macro", zero_division=0),
    }


resultados = []

for abordagem in [1, 2]:
    print(f"\n===== ABORDAGEM {abordagem} =====")
    Xtr, ytr, Xval, yval, Xte, yte = carregar(abordagem)

    # Busca de parâmetros: treina no treino, mede na validação
    tentativas = []
    for valores in itertools.product(*GRADE.values()):
        params = dict(zip(GRADE.keys(), valores))
        modelo = DecisionTreeClassifier(random_state=SEMENTE, **params).fit(Xtr, ytr)
        tentativas.append({
            **params,
            "f1_treino": f1_score(ytr, modelo.predict(Xtr), average="macro"),
            "f1_val": f1_score(yval, modelo.predict(Xval), average="macro"),
            "folhas": modelo.get_n_leaves(),
        })
    tabela = pd.DataFrame(tentativas)
    tabela.to_csv(f"arvore_busca_abordagem{abordagem}.csv", index=False)

    # Melhor F1 na validação (empate: menos folhas)
    melhor = tabela.sort_values(["f1_val", "folhas"], ascending=[False, True]).iloc[0]
    params = {k: (None if pd.isna(melhor[k]) else melhor[k]) for k in GRADE}
    if params["max_depth"] is not None:
        params["max_depth"] = int(params["max_depth"])
    params["min_samples_leaf"] = int(params["min_samples_leaf"])
    print("Melhores parâmetros:", params)
    print(f"F1 treino = {melhor.f1_treino:.3f} | F1 validação = {melhor.f1_val:.3f}")

    # Modelo final
    t0 = time.perf_counter()
    modelo = DecisionTreeClassifier(random_state=SEMENTE, **params).fit(Xtr, ytr)
    tempo_treino = (time.perf_counter() - t0) * 1000

    # Avaliação no teste
    pred = modelo.predict(Xte)
    print(classification_report(yte, pred, zero_division=0))
    resultados.append({"abordagem": abordagem, **params, **metricas(yte, pred),
                       "profundidade": modelo.get_depth(), "folhas": modelo.get_n_leaves(),
                       "tempo_treino_ms": tempo_treino})

    # Salva o modelo para o front end
    joblib.dump(modelo, f"arvore_modelo_ab{abordagem}.joblib")

# Comparação entre abordagens
resultados = pd.DataFrame(resultados)
resultados.to_csv("arvore_resultados_teste.csv", index=False)
print("\nCOMPARAÇÃO NO TESTE")
print(resultados[["abordagem", "acuracia", "precision", "recall", "f1",
                  "folhas", "tempo_treino_ms"]].round(3).to_string(index=False))
