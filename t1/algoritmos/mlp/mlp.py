from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, classification_report
from sklearn.utils import resample
import matplotlib.pyplot as plt

treino = pd.read_csv(Path(__file__).resolve().parents[2] / 'dataset' / 'treino.csv')
teste = pd.read_csv(Path(__file__).resolve().parents[2] / 'dataset' / 'teste.csv')
validacao = pd.read_csv(Path(__file__).resolve().parents[2] / 'dataset' / 'validacao.csv')

treino = treino.replace({'b': 0, 'o': -1, 'x': 1})
teste = teste.replace({'b': 0, 'o': -1, 'x': 1})
validacao = validacao.replace({'b': 0, 'o': -1, 'x': 1})

print("Antes:\n", treino['classe'].value_counts())

empate = treino[treino['classe'] == 'Empate']
outras = treino[treino['classe'] != 'Empate']

alvo = treino['classe'].value_counts().max()

empate_up = resample(empate,
                     replace=True,
                     n_samples=alvo,
                     random_state=42)

treino_bal = pd.concat([outras, empate_up]).sample(frac=1, random_state=42)

print("Depois:\n", treino_bal['classe'].value_counts())

x_treino = treino_bal.drop(columns=['classe']).astype(int)
y_treino = treino_bal['classe']
x_teste = teste.drop(columns=['classe']).astype(int)
y_teste = teste['classe']
x_validacao = validacao.drop(columns=['classe']).astype(int)
y_validacao = validacao['classe']

resultados = {}
modelos = {}
predicoes = {}

def avaliar(nome, modelo):
    modelo.fit(x_treino, y_treino)
    y_pred = modelo.predict(x_validacao)
    resultados[nome] = {
        'acuracia': accuracy_score(y_validacao, y_pred),
        'precisao': precision_score(y_validacao, y_pred, average='macro', zero_division=0),
        'recall':   recall_score(y_validacao, y_pred, average='macro', zero_division=0),
        'f1':       f1_score(y_validacao, y_pred, average='macro', zero_division=0),
    }
    modelos[nome] = modelo
    predicoes[nome] = y_pred
    return modelo, y_pred

avaliar('mlp_1',
        MLPClassifier(hidden_layer_sizes=(10,),
                   activation='relu',
                   solver='adam',
                   learning_rate_init=0.001,
                   momentum=0.1,
                   max_iter=3000,
                   random_state=42)
        )

avaliar('mlp_2',
        MLPClassifier(hidden_layer_sizes=(16,),
                   activation='relu',
                   solver='adam',
                   learning_rate_init=0.001,
                   momentum=0.1,
                   max_iter=3000,
                   random_state=42)
        )

avaliar('mlp_3',
        MLPClassifier(hidden_layer_sizes=(32,),
                   activation='relu',
                   solver='adam',
                   learning_rate_init=0.001,
                   momentum=0.1,
                   max_iter=3000,
                   random_state=42)
        )

avaliar('mlp_4',
        MLPClassifier(hidden_layer_sizes=(64,),
                   activation='relu',
                   solver='adam',
                   learning_rate_init=0.001,
                   momentum=0.1,
                   max_iter=3000,
                   random_state=42)
        )

avaliar('mlp_5',
        MLPClassifier(hidden_layer_sizes=(32,16),
                   activation='relu',
                   solver='adam',
                   learning_rate_init=0.001,
                   momentum=0.1,
                   max_iter=3000,
                   random_state=42)
        )


def graficos(nome, y_real, y_pred, conjunto):
    modelo = modelos[nome]
    rotulo = f"{nome} ({conjunto})"

    # 1) Matriz de confusão
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_real, y_pred, cmap='Blues', ax=ax)
    ax.set_title(f"Matriz de confusão – {rotulo}")
    plt.show()

    # 2) Quantidade por classe: real vs predito
    classes = np.unique(np.concatenate([np.asarray(y_real), y_pred]))
    real = [np.sum(np.asarray(y_real) == c) for c in classes]
    pred = [np.sum(y_pred == c) for c in classes]
    x = np.arange(len(classes))

    plt.figure(figsize=(8, 5))
    plt.bar(x - 0.2, real, width=0.4, label="Real", color="blue", alpha=0.7)
    plt.bar(x + 0.2, pred, width=0.4, label="Predito", color="red", alpha=0.7)
    plt.xticks(x, classes)
    plt.xlabel("Classe")
    plt.ylabel("Quantidade")
    plt.title(f"Distribuição das classes – {rotulo}")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.show()

    # 3) Curva de perda do treinamento
    plt.figure(figsize=(8, 5))
    plt.plot(modelo.loss_curve_, color="purple")
    plt.xlabel("Época")
    plt.ylabel("Perda (loss)")
    plt.title(f"Curva de aprendizado – {rotulo}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # Relatório por classe
    print(f"Relatório – {rotulo}")
    print(classification_report(y_real, y_pred, zero_division=0))

    # Estrutura da rede
    print("Camadas ocultas:", modelo.hidden_layer_sizes)
    print("Camadas totais (entrada + ocultas + saída):", len(modelo.coefs_) + 1)
    for i, (w, b) in enumerate(zip(modelo.coefs_, modelo.intercepts_)):
        print(f"Camada {i} → {i+1}: pesos {w.shape}, bias {b.shape}")


print("-----")
print('Tabela de resultados')
df = pd.DataFrame(resultados).T.round(4)
print(df.sort_values('f1', ascending=False))

# Comparação das 4 métricas na validação para cada modelo
metricas = ['acuracia', 'precisao', 'recall', 'f1']
x = np.arange(len(df))
largura = 0.2

plt.figure(figsize=(10, 5))
for i, metrica in enumerate(metricas):
    plt.bar(x + (i - 1.5) * largura, df[metrica], width=largura, label=metrica)
plt.xticks(x, df.index)
plt.ylabel("Valor")
plt.ylim(0, 1.05)
plt.title("Comparação das métricas na validação")
plt.legend()
plt.grid(True, axis="y", linestyle="--", alpha=0.6)
plt.show()

print('-----')

melhor = df['f1'].idxmax()
graficos(melhor, y_validacao, predicoes[melhor], 'validação')

# Avaliação final no TESTE
modelo_final = modelos[melhor]
y_pred_teste = modelo_final.predict(x_teste)

resultado_teste = {
    'acuracia': accuracy_score(y_teste, y_pred_teste),
    'precisao': precision_score(y_teste, y_pred_teste, average='macro', zero_division=0),
    'recall':   recall_score(y_teste, y_pred_teste, average='macro', zero_division=0),
    'f1':       f1_score(y_teste, y_pred_teste, average='macro', zero_division=0),
}

print('-----')
print(f'Resultado final no teste – {melhor}')
print(pd.Series(resultado_teste).round(4))
print('-----')

graficos(melhor, y_teste, y_pred_teste, 'teste')

# F1 no treino vs F1 na validação
print("Análise de overfitting")
overfitting = {}
for nome, modelo in modelos.items():
    y_pred_treino = modelo.predict(x_treino)
    f1_treino = f1_score(y_treino, y_pred_treino, average='macro', zero_division=0)
    f1_val = resultados[nome]['f1']
    overfitting[nome] = {
        'f1_treino': f1_treino,
        'f1_validacao': f1_val,
        'diferenca': f1_treino - f1_val,
    }

df_overfitting = pd.DataFrame(overfitting).T.round(4)
print('Análise de overfitting')
print(df_overfitting)

# Gráfico: F1 treino vs validação
x = np.arange(len(df_overfitting))
plt.figure(figsize=(9, 5))
plt.bar(x - 0.2, df_overfitting['f1_treino'], width=0.4, label="Treino", color="blue", alpha=0.7)
plt.bar(x + 0.2, df_overfitting['f1_validacao'], width=0.4, label="Validação", color="orange", alpha=0.7)
plt.xticks(x, df_overfitting.index)
plt.ylabel("F1-macro")
plt.ylim(0, 1.05)
plt.title("F1 no treino vs validação")
plt.legend()
plt.grid(True, axis="y", linestyle="--", alpha=0.6)
plt.show()

# Curvas de perda de todos os modelos juntas
plt.figure(figsize=(9, 5))
for nome, modelo in modelos.items():
    plt.plot(modelo.loss_curve_, label=nome)
plt.xlabel("Época")
plt.ylabel("Perda (loss)")
plt.title("Curvas de perda no treino")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()