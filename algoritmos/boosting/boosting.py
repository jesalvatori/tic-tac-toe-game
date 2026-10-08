# Gradient Boosting - T1 Tic Tac Toe
import itertools
import time

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.utils.class_weight import compute_sample_weight

from preprocessamento import carregar

SEMENTE = 42

# Parâmetros testados
GRADE = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.05, 0.1, 0.3],
    "max_depth": [1, 2, 3],
    "pesos": [None, "balanced"],
}


def metricas(y_real, y_pred):
    return {
        "acuracia": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_real, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_real, y_pred, average="macro", zero_division=0),
    }


def treinar(params, X, y):
    # Pesos por classe via sample_weight (o boosting não tem class_weight)
    p = dict(params)
    pesos = p.pop("pesos")
    modelo = GradientBoostingClassifier(random_state=SEMENTE, **p)
    sw = compute_sample_weight("balanced", y) if pesos == "balanced" else None
    return modelo.fit(X, y, sample_weight=sw)


resultados = []

for abordagem in [1, 2]:
    print(f"\n===== ABORDAGEM {abordagem} =====")
    Xtr, ytr, Xval, yval, Xte, yte = carregar(abordagem)

    # Busca de parâmetros: treina no treino, mede na validação
    tentativas = []
    for valores in itertools.product(*GRADE.values()):
        params = dict(zip(GRADE.keys(), valores))
        modelo = treinar(params, Xtr, ytr)
        tentativas.append({
            **params,
            "f1_treino": f1_score(ytr, modelo.predict(Xtr), average="macro"),
            "f1_val": f1_score(yval, modelo.predict(Xval), average="macro"),
        })
    tabela = pd.DataFrame(tentativas)
    tabela.to_csv(f"boosting_busca_abordagem{abordagem}.csv", index=False)

    # Melhor F1 na validação (empate: menos árvores e mais rasas)
    melhor = tabela.sort_values(["f1_val", "n_estimators", "max_depth"],
                                ascending=[False, True, True]).iloc[0]
    params = {
        "n_estimators": int(melhor.n_estimators),
        "learning_rate": float(melhor.learning_rate),
        "max_depth": int(melhor.max_depth),
        "pesos": None if pd.isna(melhor.pesos) else melhor.pesos,
    }
    print("Melhores parâmetros:", params)
    print(f"F1 treino = {melhor.f1_treino:.3f} | F1 validação = {melhor.f1_val:.3f}")

    # Modelo final
    t0 = time.perf_counter()
    modelo = treinar(params, Xtr, ytr)
    tempo_treino = (time.perf_counter() - t0) * 1000

    # Avaliação no teste
    pred = modelo.predict(Xte)
    print(classification_report(yte, pred, zero_division=0))
    resultados.append({"abordagem": abordagem, **params, **metricas(yte, pred),
                       "tempo_treino_ms": tempo_treino})

    # Salva o modelo para o front end
    joblib.dump(modelo, f"boosting_modelo_ab{abordagem}.joblib")

# Comparação entre abordagens
resultados = pd.DataFrame(resultados)
resultados.to_csv("boosting_resultados_teste.csv", index=False)
print("\nCOMPARAÇÃO NO TESTE")
print(resultados[["abordagem", "acuracia", "precision", "recall", "f1",
                  "tempo_treino_ms"]].round(3).to_string(index=False))
