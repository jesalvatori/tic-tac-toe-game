
# Gradient Boosting - T1 Tic Tac Toe
import itertools
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score
)

SEMENTE = 42
ABORDAGENS = [1, 2]

RAIZ = Path(__file__).resolve().parents[2]
PASTA_SAIDA = Path(__file__).resolve().parent

# Parâmetros testados
GRADE = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.05, 0.1, 0.3],
    "max_depth": [1, 2, 3],
    "pesos": [None],
}


def carregar(abordagem):
    # Mesmos datasets utilizados pelos demais algoritmos
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


def treinar(params, X, y):
    # O treinamento já foi balanceado com oversampling
    p = dict(params)
    p.pop("pesos")

    modelo = GradientBoostingClassifier(
        random_state=SEMENTE, **p
    )

    return modelo.fit(X, y)


# Carregamento dos dados
dados = {ab: carregar(ab) for ab in ABORDAGENS}

# Treinamento e validação
buscas, melhores_params, modelos, tempos = {}, {}, {}, {}

for ab in ABORDAGENS:
    print(f"\n===== ABORDAGEM {ab} =====")

    Xtr, ytr, Xval, yval, _, _ = dados[ab]

    print(
        f"Features: {Xtr.shape[1]} | "
        f"Treino: {len(Xtr)} | "
        f"Validação: {len(Xval)}"
    )

    # Busca de parâmetros: treina no treino, mede na validação
    tentativas = []

    for valores in itertools.product(*GRADE.values()):
        params = dict(zip(GRADE.keys(), valores))

        m = treinar(params, Xtr, ytr)

        tentativas.append({
            **params,
            "f1_treino": f1_score(
                ytr, m.predict(Xtr), average="macro"
            ),
            "f1_val": f1_score(
                yval, m.predict(Xval), average="macro"
            ),
        })

    tabela = pd.DataFrame(tentativas)

    tabela.to_csv(
        PASTA_SAIDA / f"boosting_busca_abordagem{ab}.csv",
        index=False
    )

    buscas[ab] = tabela

    # Melhor F1 na validação; em empate, o modelo mais barato
    melhor = tabela.sort_values(
        ["f1_val", "n_estimators", "max_depth"],
        ascending=[False, True, True]
    ).iloc[0]

    params = {
        "n_estimators": int(melhor.n_estimators),
        "learning_rate": float(melhor.learning_rate),
        "max_depth": int(melhor.max_depth),
        "pesos": None,
    }

    melhores_params[ab] = params

    # Modelo final (medindo o tempo de treino)
    t0 = time.perf_counter()

    modelos[ab] = treinar(params, Xtr, ytr)

    tempos[ab] = (time.perf_counter() - t0) * 1000

    print(
        f"Abordagem {ab}: {params} | "
        f"F1 treino = {melhor.f1_treino:.3f} | "
        f"F1 validação = {melhor.f1_val:.3f}"
    )


# Comparação entre as abordagens na validação
linhas = []

for ab in ABORDAGENS:
    Xtr, ytr, Xval, yval, _, _ = dados[ab]

    linhas.append({
        "abordagem": ab,
        "n_features": Xtr.shape[1],
        **metricas(yval, modelos[ab].predict(Xval)),
        "n_arvores": melhores_params[ab]["n_estimators"],
        "profundidade_arvores": melhores_params[ab]["max_depth"],
        "tempo_treino_ms": tempos[ab],
    })

comparacao_val = pd.DataFrame(linhas).set_index("abordagem")

print("\n===== COMPARAÇÃO NA VALIDAÇÃO =====")
print(comparacao_val.round(4))

# Seleção da melhor abordagem pelo F1 macro
melhor_abordagem = comparacao_val["f1"].idxmax()

print(f"\nMelhor abordagem: {melhor_abordagem}")
print(
    f"F1 macro na validação: "
    f"{comparacao_val.loc[melhor_abordagem, 'f1']:.4f}"
)


# Avaliação final somente da abordagem vencedora
ab = melhor_abordagem

_, _, _, _, Xte, yte = dados[ab]
m = modelos[ab]

t0 = time.perf_counter()

pred = m.predict(Xte)

tempo_pred = (time.perf_counter() - t0) * 1000

print(f"\n===== TESTE FINAL - ABORDAGEM {ab} =====")
print(classification_report(yte, pred, zero_division=0))

resultados_teste = pd.DataFrame([{
    "abordagem": ab,
    **melhores_params[ab],
    **metricas(yte, pred),
    "tempo_treino_ms": tempos[ab],
    "tempo_pred_ms": tempo_pred,
}])

resultados_teste.to_csv(
    PASTA_SAIDA / "boosting_resultados_teste.csv",
    index=False
)

# Salva o modelo para o front end
joblib.dump(
    m,
    PASTA_SAIDA / f"boosting_modelo_ab{ab}.joblib"
)

print("\n===== RESULTADOS FINAIS =====")
print(resultados_teste.round(4).to_string(index=False))
