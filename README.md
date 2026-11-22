# Jogo da Velha com Inteligência Artificial

Trabalhos T1 e T2 da disciplina de Inteligência Artificial, do curso de Sistemas de Informação da PUCRS.

Os dois trabalhos usam o jogo da velha, com objetivos diferentes:

| Trabalho | Função da IA | Executar | Endereço |
| --- | --- | --- | --- |
| T1 | Classificar o estado do tabuleiro | `python3 app.py` | http://127.0.0.1:5001 |
| T2 | Escolher jogadas com uma rede neural treinada por algoritmo genético | `python3 app_t2.py` | http://127.0.0.1:5002 |

## T1: classificação do tabuleiro

Compara cinco classificadores: KNN, MLP, Árvore de Decisão, Random Forest e Gradient Boosting. Cada modelo recebe o tabuleiro e prevê se ainda há jogo, se X venceu, se O venceu ou se houve empate.

O computador realiza jogadas aleatórias. Os classificadores avaliam o estado do jogo.

A documentação dos dados, experimentos e execução está em [T1.md](T1.md). O relatório está em [relatorio/relatorio_t1.md](relatorio/relatorio_t1.md).

## T2: rede neural e algoritmo genético

A rede treinada é uma **MLP (Multilayer Perceptron)**. Ela recebe o tabuleiro e produz nove saídas, uma para cada posição. A saída de maior valor indica a jogada escolhida.

| Componente | Responsabilidade |
| --- | --- |
| MLP | Escolher a jogada a partir do tabuleiro e dos pesos atuais |
| Algoritmo genético | Evoluir os pesos da MLP durante o treinamento |
| Minimax | Jogar contra a rede para avaliar os indivíduos |

A topologia inicial é 9–9–9: nove entradas, nove neurônios ocultos e nove saídas. Com os bias, cada indivíduo contém 180 parâmetros. Os pesos iniciais são valores reais aleatórios entre −1 e 1.

O treinamento usa elitismo, seleção por torneio, cruzamento BLX-α e mutação. A MLP executa apenas propagação direta, sem backpropagation.

### Modos do Minimax

- **Médio:** em cada turno, 50% de chance de usar Minimax e 50% de realizar uma jogada aleatória válida.
- **Difícil:** usa Minimax em todas as jogadas.

O treino começa no médio e depois inclui o difícil. A avaliação alterna quem começa. O AG para pelo limite de gerações ou pela estagnação da melhor aptidão na segunda fase.

### Executar o T2

Requisito: Python 3.10 ou superior.

Para baixar esta versão:

```bash
git clone --branch t2-rede-genetica https://github.com/jesalvatori/tic-tac-toe-game.git
cd tic-tac-toe-game
```

Crie e ative um ambiente virtual.

**macOS ou Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Com o ambiente ativo, instale as dependências e inicie o servidor:

```bash
python -m pip install -r requirements-t2.txt
python app_t2.py
```

Abra http://127.0.0.1:5002 no navegador. Mantenha o terminal aberto enquanto utiliza a aplicação.

### Usar a interface

O T2 mantém o estilo visual do T1. Na tela principal, escolha o adversário e quem começa a partida.

Para treinar:

1. Abra **Treinamento da rede**.
2. Defina o tamanho da população e o limite de gerações.
3. Clique em **Preparar treino**.
4. Use **Treinar** para executar automaticamente ou **Uma geração** para acompanhar o processo.
5. Consulte a evolução, a população e a partida passo a passo.
6. Escolha **Rede treinada** como adversário para jogar com os pesos encontrados.

Os demais parâmetros ficam em **Configurações avançadas**. **Pausar** conclui a geração em execução antes de interromper o treino.

Para usar o experimento incluído, clique em **Carregar rede** e selecione [t2/resultados/rede-t2.json](t2/resultados/rede-t2.json). Para guardar um novo treinamento, clique em **Salvar rede**. Os pesos ficam na memória do servidor até serem exportados.

### Notebooks por algoritmo

Cada algoritmo do T2 tem seu notebook na própria pasta, como no T1:

| Algoritmo | Notebook |
| --- | --- |
| MLP | [t2/algoritmos/mlp/mlp.ipynb](t2/algoritmos/mlp/mlp.ipynb) |
| Minimax | [t2/algoritmos/minimax/minimax.ipynb](t2/algoritmos/minimax/minimax.ipynb) |
| Algoritmo genético | [t2/algoritmos/genetico/genetico.ipynb](t2/algoritmos/genetico/genetico.ipynb) |

```bash
python3 -m pip install -r requirements-notebook-t2.txt
```

Abra o notebook no VS Code, selecione o kernel da `.venv` e execute as células na ordem. A sugestão de leitura é MLP, Minimax e depois genético. Os notebooks não dependem de variáveis de outra sessão.

A implementação de cada algoritmo aparece em células de código. Os arquivos `.py` continuam sendo usados pela interface, então uma alteração nas células não modifica automaticamente o jogo.

O notebook do genético reúne o treinamento completo e os gráficos. A exportação gera `t2/resultados/rede-notebook.json` e `t2/resultados/avaliacao-notebook.json`, preservando o experimento incluído. O jogo continua abrindo com `python3 app_t2.py`.

### Aptidão e métricas

A função de aptidão usa +3 por vitória, +1 por empate, −3 por derrota e −2 por escolha de casa ocupada. A soma é dividida pelo número de partidas.

Se a primeira escolha da rede for uma casa ocupada, o sistema registra o erro e executa a próxima saída válida. Essa recuperação permite continuar a partida, mas influencia seus resultados.

A acurácia mede a proporção de escolhas originais que pertencem ao conjunto de jogadas ótimas do Minimax. Escolhas ocupadas contam como erro. Vitórias, empates, derrotas e escolhas válidas são apresentados separadamente.

### Experimento incluído

Treino com 80 indivíduos, 250 gerações, topologia 9–9–9 e semente 42. O teste usa sementes reservadas que não participam da seleção do AG.

| Teste médio: 100 partidas | Melhor rede da primeira geração | Rede final |
| --- | ---: | ---: |
| Vitórias | 34 | 64 |
| Empates | 13 | 21 |
| Derrotas | 53 | 15 |
| Acurácia de decisões | 36,6% | 77,9% |
| Escolhas válidas | 50,8% | 82,4% |

No difícil, a rede final empatou as duas partidas avaliadas, uma em cada posição inicial. Ainda houve uma escolha inválida. Esses testes mostram melhora neste experimento e não garantem desempenho perfeito contra todas as sequências de jogadas.

Os parâmetros, o histórico e os resultados estão no [relatório do T2](t2/relatorio/relatorio_t2.md) e [t2/resultados/avaliacao.json](t2/resultados/avaliacao.json).

### Reproduzir e verificar

```bash
python -m t2.train
python -m unittest discover -s tests -v
```

O primeiro comando executa o experimento e grava os arquivos em `t2/resultados/`. O segundo verifica a lógica do jogo, Minimax, propagação, aptidão, operadores do AG e API.

## Arquivos principais

| Caminho | Conteúdo |
| --- | --- |
| `app.py` | Servidor do T1 |
| `front_end/` | Interface do T1 |
| `algoritmos/` e `dataset/` | Classificadores, experimentos e dados do T1 |
| `app_t2.py` | Servidor do T2 |
| `t2/algoritmos/mlp/` | Rede neural e propagação |
| `t2/algoritmos/minimax/` | Minimax nos modos médio e difícil |
| `t2/algoritmos/genetico/` | População e operadores do algoritmo genético |
| `t2/jogo.py` | Regras do tabuleiro |
| `t2/treinamento.py` | Partidas, recompensas e avaliação |
| `t2/relatorio/relatorio_t2.md` | Relatório do T2 |
| `t2/front_end/` | Interface do T2 |
| `t2/train.py` | Treinamento pela linha de comando |
| `t2/algoritmos/*/*.ipynb` | Notebook de cada algoritmo |
| `t2/resultados/` | Rede e resultados do experimento |
| `tests/test_t2.py` | Testes da lógica e API do T2 |

## Apoio no desenvolvimento

O projeto utilizou ChatGPT como apoio na compreensão dos conceitos, implementação, depuração, documentação e testes automatizados. A revisão da solução, a interpretação dos resultados e a apresentação fazem parte do trabalho dos integrantes do grupo.
