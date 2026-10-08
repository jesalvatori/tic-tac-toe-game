# T1 — Jogo da Velha com Inteligência Artificial

Trabalho desenvolvido para a disciplina de **Inteligência Artificial**, do curso de Sistemas de Informação da PUCRS.

## 1. Objetivo

Desenvolver, avaliar e integrar algoritmos de aprendizado de máquina capazes de classificar o estado de um tabuleiro de Jogo da Velha em quatro categorias:

- **Tem jogo:** a partida ainda não terminou.
- **X venceu:** o jogador X completou uma combinação vencedora.
- **O venceu:** o jogador O completou uma combinação vencedora.
- **Empate:** o tabuleiro está completo e não existe vencedor.

O projeto possui uma interface web interativa na qual o usuário joga contra o computador, que realiza jogadas aleatórias. Após cada jogada, o algoritmo selecionado prevê o estado do tabuleiro.

A interface permite alternar entre cinco classificadores e acompanhar o desempenho de cada um durante as partidas.

## 2. Algoritmos

Foram desenvolvidos cinco algoritmos de classificação:

| Algoritmo | Situação |
|---|---|
| K-Nearest Neighbors (KNN) | Implementado e integrado |
| Multilayer Perceptron (MLP) | Implementado e integrado |
| Árvore de Decisão | Implementado e integrado |
| Random Forest | Implementado e integrado |
| Gradient Boosting | Implementado e integrado |

Os modelos foram desenvolvidos e avaliados individualmente, com experimentos de treinamento, validação e teste.

Na integração atual com o jogo:

- **KNN:** utiliza a Abordagem 2, com K = 9.
- **MLP:** utiliza a Abordagem 2, com duas camadas ocultas de 32 e 16 neurônios.
- **Árvore de Decisão:** utiliza o modelo salvo da Abordagem 2.
- **Random Forest:** utiliza a configuração `rf_3`, treinada com o dataset original.
- **Gradient Boosting:** utiliza o modelo salvo da Abordagem 2.

A configuração da Random Forest deve ser conferida com a versão final dos experimentos antes da entrega.

## 3. Dataset

Os dados originais estão divididos em três arquivos:

- `treino.csv`: treinamento dos modelos.
- `validacao.csv`: seleção de parâmetros e comparação de configurações.
- `teste.csv`: avaliação final dos modelos.

Os arquivos originais estão localizados na pasta `dataset/`.

A partir desses dados, foram desenvolvidas duas abordagens de representação dos tabuleiros.

### 3.1. Abordagem 1 — Representação das casas

Utiliza nove atributos, correspondentes às nove posições do tabuleiro.

Os símbolos são convertidos em valores numéricos:

- X = 1
- O = -1
- Casa vazia = 0

**Script:** `dataset/abordagem_1/preparar_a1.py`

**Arquivos gerados:** `dataset/abordagem_1/`

### 3.2. Abordagem 2 — Características derivadas

Utiliza quinze características calculadas a partir do tabuleiro:

- Quantidade de X.
- Quantidade de O.
- Ocupação das nove casas.
- Quantidade de linhas com dois X.
- Quantidade de linhas com dois O.
- Quantidade de casas vazias.
- Indicador do próximo jogador.

**Script:** `dataset/abordagem_2/preparar_a2.py`

**Arquivos gerados:** `dataset/abordagem_2/`

Em ambas as abordagens, o balanceamento das classes é aplicado somente ao conjunto de treinamento.

A integração atual da Random Forest utiliza diretamente o dataset original, conforme o experimento selecionado para esse algoritmo.

## 4. Experimentos e avaliação

Cada algoritmo possui uma pasta específica em `algoritmos/`, contendo seu código de treinamento e notebook de experimentos.

Os experimentos incluem, conforme a implementação de cada algoritmo:

- Treinamento de diferentes configurações.
- Seleção de parâmetros com dados de validação.
- Comparação entre modelos.
- Avaliação de desempenho.
- Análise dos resultados obtidos.

As principais métricas utilizadas são:

- **Acurácia:** proporção de classificações corretas.
- **Precisão:** proporção de previsões positivas corretas para cada classe.
- **Recall:** capacidade de identificar corretamente os exemplos de cada classe.
- **F1-score:** média harmônica entre precisão e recall.

O F1-score macro permite avaliar o desempenho considerando todas as classes.

A análise e a comparação entre os classificadores são documentadas em `relatorio/relatorio_t1.md`.

## 5. Interface do jogo

A interface foi desenvolvida utilizando:

- **HTML:** estrutura da página.
- **CSS:** apresentação visual e layout responsivo.
- **JavaScript:** lógica das partidas e comunicação com o servidor.
- **Flask (Python):** integração entre o Front End e os classificadores.

O usuário controla o jogador **X**, enquanto o computador controla o jogador **O**, realizando jogadas aleatórias.

Os algoritmos de aprendizado de máquina não escolhem as jogadas do computador. Sua função é classificar o estado atual do tabuleiro.

### 5.1. Seleção dos algoritmos

O usuário pode selecionar qualquer um dos cinco algoritmos disponíveis:

1. KNN
2. MLP
3. Árvore de Decisão
4. Random Forest
5. Gradient Boosting

A classificação é realizada pelo algoritmo selecionado na interface.

### 5.2. Integração com os modelos

A comunicação entre o Front End e o Back End acontece da seguinte forma:

1. O usuário ou o computador realiza uma jogada.
2. O JavaScript envia o estado do tabuleiro e o algoritmo selecionado ao servidor Flask.
3. O Flask converte o tabuleiro para a representação utilizada pelo modelo.
4. O classificador realiza a previsão.
5. O servidor retorna o resultado ao JavaScript.
6. A interface compara a previsão com o estado real.
7. Os indicadores de desempenho são atualizados.

A comunicação é realizada por meio da rota `POST /api/prever`.

O servidor também disponibiliza a rota `GET /api/modelos`, utilizada para verificar os algoritmos carregados.

### 5.3. Informações exibidas

A interface apresenta:

- Tabuleiro interativo de Jogo da Velha.
- Seleção do algoritmo de classificação.
- Estado real do tabuleiro.
- Previsão realizada pela IA.
- Quantidade de acertos da IA.
- Quantidade de erros da IA.
- Acurácia acumulada da IA.
- Quantidade de vitórias do jogador.
- Quantidade de vitórias do computador.
- Quantidade de empates.
- Botão para iniciar uma nova partida.

Os indicadores de desempenho dos classificadores são mantidos separadamente por algoritmo.

### 5.4. Cálculo da acurácia

A acurácia exibida durante o jogo é calculada por:

**Acurácia = Acertos / (Acertos + Erros) × 100**

Essa acurácia corresponde às classificações realizadas durante as partidas interativas.

Ela não deve ser confundida com a acurácia obtida nos experimentos utilizando o conjunto de teste.

### 5.5. Regras do jogo

A classificação ocorre após cada jogada, inclusive nas jogadas do computador.

Se o classificador indicar incorretamente que a partida terminou, o erro será contabilizado, mas o jogo continuará normalmente.

Se houver um resultado final real, a partida será encerrada conforme as regras do Jogo da Velha, independentemente da previsão realizada pelo algoritmo.

## 6. Estrutura do projeto

O projeto está organizado nas seguintes pastas e arquivos principais:

- **`algoritmos/`**
  - `knn/`: implementação e experimentos do KNN.
  - `mlp/`: implementação e experimentos da MLP.
  - `arvore_decisao/`: implementação e modelo da Árvore de Decisão.
  - `random_forest/`: implementação e experimentos da Random Forest.
  - `boosting/`: implementação e modelo do Gradient Boosting.
- **`dataset/`**
  - `abordagem_1/`: dados e preparação da primeira abordagem.
  - `abordagem_2/`: dados e preparação da segunda abordagem.
  - `treino.csv`
  - `validacao.csv`
  - `teste.csv`
- **`front_end/`**
  - `index.html`: estrutura da interface.
  - `style.css`: estilos e layout responsivo.
  - `script.js`: funcionamento do jogo e comunicação com Flask.
- **`relatorio/`**
  - `relatorio_t1.md`: relatório dos experimentos e resultados.
- **`app.py`**: servidor Flask e integração dos classificadores.
- **`README.md`**: documentação do projeto.
- **`.gitignore`**: arquivos e diretórios ignorados pelo Git.

## 7. Como executar o projeto

### 7.1. Requisitos

- Python 3
- Flask
- Pandas
- Scikit-learn
- Joblib
- Matplotlib e Jupyter para executar os experimentos
- Navegador web

### 7.2. Clonar o repositório

Execute:

`git clone https://github.com/jesalvatori/tic-tac-toe-game.git`

Entre na pasta do projeto:

`cd tic-tac-toe-game`

### 7.3. Criar e ativar o ambiente virtual

**macOS ou Linux:**

Criar o ambiente:

`python3 -m venv .venv`

Ativar:

`source .venv/bin/activate`

**Windows:**

Criar o ambiente:

`python -m venv .venv`

Ativar:

`.venv\Scripts\activate`

### 7.4. Instalar as dependências

Execute:

`python -m pip install flask pandas scikit-learn joblib matplotlib jupyter`

### 7.5. Gerar os dados das abordagens (opcional)

Os arquivos processados já estão disponíveis no projeto.

Caso seja necessário gerá-los novamente, execute:

`python dataset/abordagem_1/preparar_a1.py`

`python dataset/abordagem_2/preparar_a2.py`

### 7.6. Iniciar o servidor Flask

Na pasta raiz do projeto, execute:

`python app.py`

Aguarde o carregamento dos cinco modelos.

Na configuração atual, o servidor utiliza o endereço:

**http://127.0.0.1:5001**

### 7.7. Verificar os modelos

Para consultar os modelos carregados, acesse:

**http://127.0.0.1:5001/api/modelos**

A resposta apresenta os cinco algoritmos disponíveis no servidor.

### 7.8. Jogar

1. Abra o jogo no navegador.
2. Selecione um dos cinco algoritmos.
3. Clique em uma casa vazia para marcar X.
4. Aguarde a jogada aleatória do computador.
5. Observe a previsão realizada pelo classificador.
6. Acompanhe os acertos, erros e a acurácia da IA.
7. Clique em **Nova partida** para jogar novamente.
8. Selecione outro algoritmo para comparar o comportamento das classificações.

## 8. Situação atual e etapas finais

### Funcionalidades implementadas

- [x] Preparação dos datasets.
- [x] Implementação das duas abordagens de representação.
- [x] Desenvolvimento dos cinco algoritmos.
- [x] Desenvolvimento da interface interativa.
- [x] Integração do KNN ao Flask.
- [x] Integração da MLP ao Flask.
- [x] Integração da Árvore de Decisão ao Flask.
- [x] Integração da Random Forest ao Flask.
- [x] Integração do Gradient Boosting ao Flask.
- [x] Seleção dos algoritmos pela interface.
- [x] Exibição das previsões e indicadores de desempenho.
- [x] Ajuste do layout para diferentes tamanhos de tela.

### Etapas de validação e entrega

- [ ] Confirmar a configuração definitiva da Random Forest.
- [ ] Validar as previsões dos cinco algoritmos durante as partidas.
- [ ] Consolidar a comparação dos resultados experimentais.
- [ ] Revisar e finalizar o relatório do grupo.
- [ ] Enviar e verificar a versão final no GitHub.

## 9. Repositório

https://github.com/jesalvatori/tic-tac-toe-game

