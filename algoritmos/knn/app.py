
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from algoritmos.knn.knn import criar_modelo
from dataset.abordagem_2.preparar_a2 import extrair_caracteristicas


RAIZ = Path(__file__).resolve().parent

app = Flask(
    __name__,
    static_folder=str(RAIZ / "front_end"),
    static_url_path=""
)

# Modelos disponíveis para integração
MODELOS = {
    "knn": None,
    "mlp": None,
    "arvore": None,
    "random_forest": None,
    "boosting": None,
}

# Carregar o treinamento da Abordagem 2
treino = pd.read_csv(
    RAIZ / "dataset" / "abordagem_2" / "treino.csv"
)

X_treino = treino.drop(columns=["classe"])
y_treino = treino["classe"]

# Treinar o KNN com a configuração selecionada
modelo_knn = criar_modelo(9)
modelo_knn.fit(X_treino, y_treino)

MODELOS["knn"] = modelo_knn


@app.get("/")
def pagina_inicial():
    return send_from_directory(
        app.static_folder,
        "index.html"
    )


@app.post("/api/prever")
def prever():
    dados = request.get_json(silent=True) or {}

    algoritmo = dados.get("algoritmo")
    tabuleiro = dados.get("tabuleiro")

    if algoritmo not in MODELOS:
        return jsonify(
            erro="Algoritmo desconhecido."
        ), 400

    if MODELOS[algoritmo] is None:
        return jsonify(
            erro="Este algoritmo ainda não foi integrado."
        ), 501

    if (
        not isinstance(tabuleiro, list)
        or len(tabuleiro) != 9
        or any(casa not in ("X", "O", "") for casa in tabuleiro)
    ):
        return jsonify(
            erro="Tabuleiro inválido."
        ), 400

    # Converter o formato do JavaScript para o dataset
    tabuleiro_convertido = [
        casa.lower() if casa else "b"
        for casa in tabuleiro
    ]

    # Extrair as mesmas características usadas no treino
    caracteristicas = extrair_caracteristicas(
        tabuleiro_convertido
    )

    entrada = pd.DataFrame(
        [caracteristicas],
        columns=X_treino.columns
    )

    resultado = MODELOS[algoritmo].predict(entrada)[0]

    return jsonify(
        previsao=str(resultado),
        algoritmo=algoritmo
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
