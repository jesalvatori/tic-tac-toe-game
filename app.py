
from pathlib import Path

import joblib
import pandas as pd

from flask import Flask, jsonify, request, send_from_directory

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from algoritmos.knn.knn import criar_modelo
from dataset.abordagem_2.preparar_a2 import extrair_caracteristicas


# ==================================================
# 1. CONFIGURACAO DO PROJETO
# ==================================================

RAIZ = Path(__file__).resolve().parent

app = Flask(
    __name__,
    static_folder=str(RAIZ / "front_end"),
    static_url_path=""
)

VERSAO = "integracao-5-algoritmos"


# ==================================================
# 2. CARREGAMENTO DOS DATASETS
# ==================================================

def carregar_treino_abordagem_2():

    caminho = (
        RAIZ
        / "dataset"
        / "abordagem_2"
        / "treino.csv"
    )

    dados = pd.read_csv(caminho)

    X = dados.drop(columns=["classe"])
    y = dados["classe"]

    return X, y


def carregar_treino_original():

    caminho = RAIZ / "dataset" / "treino.csv"

    dados = pd.read_csv(caminho)

    # Codificacao usada pela Random Forest:
    # X = 1
    # O = -1
    # Vazio = 0

    dados = dados.replace({
        "x": 1,
        "o": -1,
        "b": 0
    })

    X = dados.drop(columns=["classe"])
    y = dados["classe"]

    return X, y


X_treino_a2, y_treino_a2 = carregar_treino_abordagem_2()

X_treino_original, y_treino_original = (
    carregar_treino_original()
)


# ==================================================
# 3. INTEGRACAO DOS CINCO ALGORITMOS
# ==================================================

MODELOS = {}


# --------------------------------------------------
# 3.1 KNN
# --------------------------------------------------

print("Carregando KNN...", flush=True)

modelo_knn = criar_modelo(9)

modelo_knn.fit(
    X_treino_a2,
    y_treino_a2
)

MODELOS["knn"] = modelo_knn


# --------------------------------------------------
# 3.2 MULTILAYER PERCEPTRON (MLP)
# --------------------------------------------------

print("Carregando MLP...", flush=True)

modelo_mlp = Pipeline([

    (
        "normalizacao",
        StandardScaler()
    ),

    (
        "mlp",
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=3000,
            random_state=42
        )
    )

])

modelo_mlp.fit(
    X_treino_a2,
    y_treino_a2
)

MODELOS["mlp"] = modelo_mlp


# --------------------------------------------------
# 3.3 ARVORE DE DECISAO
# --------------------------------------------------

print("Carregando Arvore de Decisao...", flush=True)

caminho_arvore = (
    RAIZ
    / "algoritmos"
    / "arvore_decisao"
    / "arvore_modelo_ab2.joblib"
)

if not caminho_arvore.exists():
    raise FileNotFoundError(
        f"Modelo da Arvore nao encontrado: {caminho_arvore}"
    )

MODELOS["arvore"] = joblib.load(caminho_arvore)


# --------------------------------------------------
# 3.4 RANDOM FOREST
# --------------------------------------------------

print("Carregando Random Forest...", flush=True)

# Configuracao rf_3 do codigo consultado
# Treinamento com o dataset original

modelo_rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

modelo_rf.fit(
    X_treino_original,
    y_treino_original
)

MODELOS["random_forest"] = modelo_rf


# --------------------------------------------------
# 3.5 GRADIENT BOOSTING
# --------------------------------------------------

print("Carregando Gradient Boosting...", flush=True)

caminho_boosting = (
    RAIZ
    / "algoritmos"
    / "boosting"
    / "boosting_modelo_ab2.joblib"
)

if not caminho_boosting.exists():
    raise FileNotFoundError(
        f"Modelo Boosting nao encontrado: {caminho_boosting}"
    )

MODELOS["boosting"] = joblib.load(caminho_boosting)


# ==================================================
# 4. CONFIRMACAO DOS MODELOS
# ==================================================

print("\n==================================", flush=True)
print("MODELOS INTEGRADOS AO JOGO", flush=True)
print("==================================", flush=True)

for nome, modelo in MODELOS.items():

    print(
        f"{nome}: {type(modelo).__name__}",
        flush=True
    )

print("==================================\n", flush=True)


# ==================================================
# 5. PAGINA INICIAL
# ==================================================

@app.get("/")
def pagina_inicial():

    return send_from_directory(
        app.static_folder,
        "index.html"
    )


# ==================================================
# 6. API PARA VERIFICAR OS MODELOS
# ==================================================

@app.get("/api/modelos")
def listar_modelos():

    return jsonify({
        "versao": VERSAO,
        "quantidade": len(MODELOS),
        "algoritmos": list(MODELOS.keys()),
        "status": "integrados"
    })


# ==================================================
# 7. CONVERSAO DO TABULEIRO
# ==================================================

def preparar_entrada(tabuleiro, algoritmo):

    # Recebe do JavaScript:
    # ["X", "O", "", "", ...]
    #
    # Converte para:
    # ["x", "o", "b", "b", ...]

    tabuleiro_convertido = [
        casa.lower() if casa else "b"
        for casa in tabuleiro
    ]

    # Random Forest usa as nove posicoes
    # codificadas numericamente

    if algoritmo == "random_forest":

        codificacao = {
            "x": 1,
            "o": -1,
            "b": 0
        }

        valores = [
            codificacao[casa]
            for casa in tabuleiro_convertido
        ]

        entrada = pd.DataFrame(
            [valores],
            columns=X_treino_original.columns
        )

    # Demais algoritmos usam Abordagem 2

    else:

        caracteristicas = extrair_caracteristicas(
            tabuleiro_convertido
        )

        entrada = pd.DataFrame(
            [caracteristicas],
            columns=X_treino_a2.columns
        )

    return entrada


# ==================================================
# 8. API DE PREVISAO
# ==================================================

@app.post("/api/prever")
def prever():

    dados = request.get_json(silent=True)

    if not isinstance(dados, dict):

        return jsonify(
            erro="Envie um JSON valido."
        ), 400

    algoritmo = dados.get("algoritmo")
    tabuleiro = dados.get("tabuleiro")

    # Verifica se o algoritmo existe

    if algoritmo not in MODELOS:

        return jsonify(
            erro="Algoritmo desconhecido."
        ), 400

    # Verifica se o tabuleiro e valido

    if (
        not isinstance(tabuleiro, list)
        or len(tabuleiro) != 9
        or any(
            casa not in ("X", "O", "")
            for casa in tabuleiro
        )
    ):

        return jsonify(
            erro="Tabuleiro invalido."
        ), 400

    # Prepara os dados de entrada

    entrada = preparar_entrada(
        tabuleiro,
        algoritmo
    )

    # Seleciona o modelo escolhido no Front End

    modelo = MODELOS[algoritmo]

    # Realiza a classificacao

    resultado = modelo.predict(entrada)[0]

    print(
        f"Algoritmo: {algoritmo} | Previsao: {resultado}",
        flush=True
    )

    # Envia a resposta para o JavaScript

    return jsonify(
        previsao=str(resultado),
        algoritmo=algoritmo
    )


# ==================================================
# 9. EXECUCAO DO SERVIDOR
# ==================================================

if __name__ == "__main__":

    print(
        "\nAcesse: http://127.0.0.1:5001",
        flush=True
    )

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        use_reloader=False
    )
