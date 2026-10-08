"""
T1 - Tic Tac Toe com ML | Etapa 4: divisão física do dataset

Entrada : dataset_balanceado.csv
Saída   : treino.csv (70%), validacao.csv (15%), teste.csv (15%)

A divisão é ESTRATIFICADA: cada conjunto mantém a mesma proporção de classes
do dataset original.

"""
import pandas as pd
from sklearn.model_selection import train_test_split

SEMENTE = 42

df = pd.read_csv("dataset_balanceado.csv")

# 1º corte: separa 15% para teste
resto, teste = train_test_split(
    df, test_size=0.15, stratify=df["classe"], random_state=SEMENTE)

# 2º corte: dos 85% restantes, separa 15% do total para validação
# (0.15 / 0.85 ≈ 0.1765)
treino, validacao = train_test_split(
    resto, test_size=0.15 / 0.85, stratify=resto["classe"], random_state=SEMENTE)

for nome, parte in [("treino", treino), ("validacao", validacao), ("teste", teste)]:
    parte.to_csv(f"{nome}.csv", index=False)

# Tabela de conferência
resumo = pd.DataFrame({
    "treino": treino["classe"].value_counts(),
    "validacao": validacao["classe"].value_counts(),
    "teste": teste["classe"].value_counts(),
})
resumo.loc["Total"] = resumo.sum()
print(resumo)

# Garantia de que nenhum tabuleiro aparece em dois conjuntos
cols = list(df.columns[:-1])
chave = lambda d: set(map(tuple, d[cols].values))
assert not (chave(treino) & chave(validacao)), "vazamento treino/validação"
assert not (chave(treino) & chave(teste)), "vazamento treino/teste"
assert not (chave(validacao) & chave(teste)), "vazamento validação/teste"
print("\nOK: nenhum tabuleiro repetido entre os conjuntos.")
