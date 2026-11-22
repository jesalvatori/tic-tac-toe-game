# Jogo da Velha com Inteligência Artificial

Trabalhos da disciplina de Inteligência Artificial da PUCRS. Cada trabalho tem sua própria pasta, com os algoritmos, a interface e a documentação.

| Trabalho | Objetivo | Documentação |
| --- | --- | --- |
| T1 | Classificar o estado do tabuleiro com cinco algoritmos | [t1/README.md](t1/README.md) |
| T2 | Aprender a escolher jogadas com MLP, algoritmo genético e Minimax | [t2/README.md](t2/README.md) |

## Organização

| Pasta | Conteúdo |
| --- | --- |
| `t1/algoritmos/` | KNN, MLP, Árvore de Decisão, Random Forest e Boosting, com seus notebooks e arquivos Python |
| `t1/dataset/` | Dados de treinamento, validação e teste do T1 |
| `t1/front_end/` | Interface do T1 |
| `t1/relatorio/` | Relatório do T1 |
| `t2/algoritmos/` | MLP, Minimax e genético, cada um com código, notebook, gráficos PNG e tabelas CSV |
| `t2/front_end/` | Interface do T2 |
| `t2/relatorio/` | Relatório do T2 |
| `t2/apoio/` | Regras do jogo, avaliação das redes, treino pelo terminal e testes |

## Abrir os jogos

Execute os comandos na raiz do repositório. Primeiro crie e ative o ambiente virtual, se ainda não tiver um:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

No Windows, use `python -m venv .venv` e `.venv\Scripts\Activate.ps1` no PowerShell.

### T1

```bash
python3 -m pip install -r t1/requirements.txt
python3 -m t1.app
```

Acesse http://127.0.0.1:5001. Aguarde o carregamento dos classificadores.

### T2

```bash
python3 -m pip install -r t2/requirements.txt
python3 -m t2.app
```

Acesse http://127.0.0.1:5002. Para jogar com o experimento incluído, carregue `t2/algoritmos/genetico/rede-t2.json` na interface.

## Notebooks do T2

| Algoritmo | Notebook |
| --- | --- |
| MLP | [t2/algoritmos/mlp/mlp.ipynb](t2/algoritmos/mlp/mlp.ipynb) |
| Minimax | [t2/algoritmos/minimax/minimax.ipynb](t2/algoritmos/minimax/minimax.ipynb) |
| Genético | [t2/algoritmos/genetico/genetico.ipynb](t2/algoritmos/genetico/genetico.ipynb) |

Instale as dependências com `python3 -m pip install -r t2/apoio/requirements-notebook.txt`. Abra o notebook no VS Code, escolha o kernel da `.venv` e execute as células na ordem. O treino completo fica no notebook do genético. Os gráficos e as tabelas também são salvos como PNG e CSV na pasta de cada algoritmo, junto do notebook, como no T1.

Os arquivos `.py` são os usados pelo jogo. Uma alteração nas células do notebook não modifica automaticamente o módulo Python correspondente.

## Treinamento e testes do T2

```bash
python3 -m t2.apoio.train
python3 -m unittest discover -s t2/apoio/tests -v
```

A configuração inicial usa 80 indivíduos, topologia 9–9–9, 250 gerações e semente 42. Os valores, as recompensas e os resultados do experimento estão no [relatório do T2](t2/relatorio/relatorio_t2.md).

## Apoio no desenvolvimento

Durante o desenvolvimento do T1 e do T2, usamos o ChatGPT (OpenAI) como ferramenta de apoio para esclarecer conceitos dos algoritmos, ajudar na implementação e revisão de trechos de código e na correção de erros. Também utilizamos a ferramenta em dúvidas sobre a integração com a interface do jogo, a interpretação das métricas e dos gráficos e a organização da documentação. As decisões e os ajustes foram discutidos pelo grupo, que realizou os experimentos e conferiu os resultados e o funcionamento da aplicação.