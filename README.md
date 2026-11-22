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
| `t2/algoritmos/` | MLP, Minimax e genético, cada um com seu notebook e arquivo Python |
| `t2/front_end/` | Interface do T2 |
| `t2/resultados/` | Rede treinada e avaliação do experimento |
| `t2/relatorio/` | Relatório do T2 |
| `t2/tests/` | Testes da lógica e da API |
| `t2/scripts/` | Verificação automática da interface |

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

Acesse http://127.0.0.1:5002. Para jogar com o experimento incluído, carregue `t2/resultados/rede-t2.json` na interface.

## Notebooks do T2

| Algoritmo | Notebook |
| --- | --- |
| MLP | [t2/algoritmos/mlp/mlp.ipynb](t2/algoritmos/mlp/mlp.ipynb) |
| Minimax | [t2/algoritmos/minimax/minimax.ipynb](t2/algoritmos/minimax/minimax.ipynb) |
| Genético | [t2/algoritmos/genetico/genetico.ipynb](t2/algoritmos/genetico/genetico.ipynb) |

Instale as dependências com `python3 -m pip install -r t2/requirements-notebook.txt`. Abra o notebook no VS Code, escolha o kernel da `.venv` e execute as células na ordem. O treino completo fica no notebook do genético.

Os arquivos `.py` são os usados pelo jogo. Uma alteração nas células do notebook não modifica automaticamente o módulo Python correspondente.

## Treinamento e testes do T2

```bash
python3 -m t2.train
python3 -m unittest discover -s t2/tests -v
```

A configuração inicial usa 80 indivíduos, topologia 9–9–9, 250 gerações e semente 42. Os valores, as recompensas e os resultados do experimento estão no [relatório do T2](t2/relatorio/relatorio_t2.md).

## Apoio no desenvolvimento

O projeto utilizou ChatGPT como apoio na compreensão dos conceitos, implementação, depuração, documentação e testes automatizados. A revisão da solução, a interpretação dos resultados e a apresentação fazem parte do trabalho dos integrantes do grupo.
