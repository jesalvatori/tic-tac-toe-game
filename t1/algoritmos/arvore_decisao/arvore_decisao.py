
# Árvore de Decisão - T1 Tic Tac Toe
import itertools
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score
)
from sklearn.tree import DecisionTreeClassifier

SEMENTE = 42
RAIZ = Path(__file__).resolve().parents[2]
PASTA_SAIDA = Path(__file__).resolve().parent

# Parâmetros testados
GRADE = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
    "min_samples_leaf": [1, 2, 5, 10],
    "class_weight": [None],
}


def carregar(abordagem):
    pasta = RAIZ / "dataset" / f"abordagem_{abordagem}"
    saida = []

    for nome in ["treino", "validacao", "teste"]:
        df = pd.read_csv(pasta / f"{nome}.csv")
        X = df.drop(columns=["classe"])
        y = df["classe"]
        saida.extend([X, y])

    return saida


def metricas(y_real, y_pred):
    return {
        "acuracia": accuracy_score(y_real, y_pred),
        "precision": precision_score(
            y_real, y_pred, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_real, y_pred, average="macro", zero_division=0
        ),
        "f1": f1_score(
            y_real, y_pred, average="macro", zero_division=0
        ),
    }


resultados_validacao = []
modelos = {}
dados_teste = {}

for abordagem in [1, 2]:
    print(f"\n===== ABORDAGEM {abordagem} =====")

    Xtr, ytr, Xval, yval, Xte, yte = carregar(abordagem)

    # Busca de parâmetros: treina no treino, mede na validação
    tentativas = []

    for valores in itertools.product(*GRADE.values()):
        params = dict(zip(GRADE.keys(), valores))
        modelo = DecisionTreeClassifier(
            random_state=SEMENTE, **params
        ).fit(Xtr, ytr)

        tentativas.append({
            **params,
            "f1_treino": f1_score(
                ytr, modelo.predict(Xtr), average="macro"
            ),
            "f1_val": f1_score(
                yval, modelo.predict(Xval), average="macro"
            ),
            "folhas": modelo.get_n_leaves(),
        })

    tabela = pd.DataFrame(tentativas)
    tabela.to_csv(
        PASTA_SAIDA / f"arvore_busca_abordagem{abordagem}.csv",
        index=False
    )

    # Melhor F1 na validação (empate: menos folhas)
    melhor = tabela.sort_values(
        ["f1_val", "folhas"], ascending=[False, True]
    ).iloc[0]

    params = {
        k: (None if pd.isna(melhor[k]) else melhor[k])
        for k in GRADE
    }

    if params["max_depth"] is not None:
        params["max_depth"] = int(params["max_depth"])

    params["min_samples_leaf"] = int(params["min_samples_leaf"])

    print("Melhores parâmetros:", params)
    print(
        f"F1 treino = {melhor.f1_treino:.3f} | "
        f"F1 validação = {melhor.f1_val:.3f}"
    )

    # Modelo final
    t0 = time.perf_counter()
    modelo = DecisionTreeClassifier(
        random_state=SEMENTE, **params
    ).fit(Xtr, ytr)

    tempo_treino = (time.perf_counter() - t0) * 1000

    modelos[abordagem] = modelo
    dados_teste[abordagem] = (Xte, yte)

    resultados_validacao.append({
        "abordagem": abordagem,
        **params,
        **metricas(yval, modelo.predict(Xval)),
        "profundidade": modelo.get_depth(),
        "folhas": modelo.get_n_leaves(),
        "tempo_treino_ms": tempo_treino,
    })


# Comparação entre abordagens na validação
comparacao_val = pd.DataFrame(resultados_validacao)

print("\nCOMPARAÇÃO NA VALIDAÇÃO")
print(comparacao_val[
    ["abordagem", "acuracia", "precision", "recall", "f1", "folhas"]
].round(4).to_string(index=False))

# Escolha da melhor abordagem pelo F1 macro
melhor_abordagem = int(
    comparacao_val.loc[comparacao_val["f1"].idxmax(), "abordagem"]
)

print(f"\nMelhor abordagem: {melhor_abordagem}")

# Avaliação final somente da abordagem vencedora
modelo_final = modelos[melhor_abordagem]
Xte, yte = dados_teste[melhor_abordagem]

pred = modelo_final.predict(Xte)

print("\nRESULTADOS NO TESTE")
print(classification_report(yte, pred, zero_division=0))

resultados_teste = pd.DataFrame([{
    "abordagem": melhor_abordagem,
    **metricas(yte, pred),
    "profundidade": modelo_final.get_depth(),
    "folhas": modelo_final.get_n_leaves(),
}])

resultados_teste.to_csv(
    PASTA_SAIDA / "arvore_resultados_teste.csv",
    index=False
)

# Salva o modelo para integração com o jogo
joblib.dump(
    modelo_final,
    PASTA_SAIDA / f"arvore_modelo_ab{melhor_abordagem}.joblib"
)

print(resultados_teste.round(4).to_string(index=False))
