from pathlib import Path
import pandas as pd

# Abordagem 1
# Representação numérica das nove casas do tabuleiro

PASTA = Path(__file__).resolve().parent
DATASET = PASTA.parent

COLUNAS = [
    "tl", "tm", "tr",
    "ml", "mm", "mr",
    "bl", "bm", "br",
]

CODIFICACAO = {"x": 1, "o": -1, "b": 0}

SEMENTE = 42


def preparar_abordagem_1(dados):
    """Converte as nove posições em valores numéricos."""
    resultado = dados.copy()

    for coluna in COLUNAS:
        resultado[coluna] = resultado[coluna].map(CODIFICACAO)

    if resultado[COLUNAS].isna().any().any():
        raise ValueError("Há valores inválidos no tabuleiro.")

    return resultado


def balancear_treino(dados):
    """Equilibra as classes somente no conjunto de treino."""
    maior_classe = dados["classe"].value_counts().max()
    partes = []

    for _, grupo in dados.groupby("classe"):
        partes.append(
            grupo.sample(
                n=maior_classe,
                replace=len(grupo) < maior_classe,
                random_state=SEMENTE,
            )
        )

    return (
        pd.concat(partes)
        .sample(frac=1, random_state=SEMENTE)
        .reset_index(drop=True)
    )


def executar():
    for nome in ["treino", "validacao", "teste"]:
        # Leitura do conjunto já dividido
        entrada = DATASET / f"{nome}.csv"
        dados = pd.read_csv(entrada)

        # Balanceamento apenas do treino
        if nome == "treino":
            dados = balancear_treino(dados)

        # Aplicação da Abordagem 1
        dados_a1 = preparar_abordagem_1(dados)

        # Exportação
        saida = PASTA / f"{nome}.csv"
        dados_a1.to_csv(saida, index=False)

        print(f"\n{nome.upper()}")
        print(f"Registros: {len(dados_a1)}")
        print(dados_a1["classe"].value_counts())

    print("\nAbordagem 1 concluída!")


if __name__ == "__main__":
    executar()