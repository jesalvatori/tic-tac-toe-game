# T2 — Jogo da Velha

A documentação completa está em [relatorio/relatorio_t2.md](relatorio/relatorio_t2.md), junto da parte do relatório, como na organização do T1.

Para abrir o jogo, execute na raiz do repositório:

```bash
python3 -m t2.app
```

Acesse http://127.0.0.1:5002. Os algoritmos ficam em `t2/algoritmos/`, a interface em `t2/front_end/` e os arquivos do experimento junto do genético, em `t2/algoritmos/genetico/`.

Os notebooks ficam junto de cada algoritmo:

- [MLP](algoritmos/mlp/mlp.ipynb)
- [Minimax](algoritmos/minimax/minimax.ipynb)
- [Algoritmo genético](algoritmos/genetico/genetico.ipynb)

Abra no VS Code e selecione o kernel da `.venv`. As dependências estão em `t2/apoio/requirements-notebook.txt`.

Cada notebook tem as explicações, o código e as saídas do próprio algoritmo. Os gráficos já estão salvos no arquivo; para refazer os experimentos, execute as células na ordem com **Run All**.

| Notebook | O que observar |
| --- | --- |
| MLP | Escores das nove casas, mapas dos pesos, distribuição dos parâmetros e comparação entre uma rede aleatória e a rede treinada. |
| Minimax | Valor de cada jogada disponível, frequência das escolhas no médio e no difícil e partidas contra um jogador aleatório. |
| Genético | Aptidão e jogadas válidas ao longo das gerações, resultados das partidas, diversidade da população, aptidões finais e comparação entre a primeira e a última geração. |

No genético, a linha tracejada marca a entrada do modo difícil no treino. A avaliação passa a incluir mais duas partidas nessa fase, então os resultados ao longo das gerações aparecem em percentuais. A diversidade é a média dos desvios padrão de cada gene na população; ela ajuda a ver se os indivíduos estão ficando parecidos.

As comparações extras da MLP e do Minimax são demonstrações dos notebooks. Os resultados do experimento incluído ficam em `algoritmos/genetico/rede-t2.json` e `algoritmos/genetico/avaliacao.json`. Executar o notebook genético salva novos arquivos com o sufixo `-notebook`, sem sobrescrever esses dois.

## Organização como no T1

Cada pasta de algoritmo reúne o notebook, o arquivo Python e as saídas do experimento. Os gráficos ficam em PNG e as tabelas em CSV. Dá para conferir o resultado no próprio notebook ou abrir o arquivo pela pasta. Ao executar o notebook de novo, ele atualiza essas saídas.

| Pasta | Arquivos |
| --- | --- |
| `algoritmos/mlp/` | `mlp.ipynb`, `mlp.py`, mapas dos pesos, escores e comparações em PNG e CSV |
| `algoritmos/minimax/` | `minimax.ipynb`, `minimax.py`, escolhas e resultados em PNG e CSV |
| `algoritmos/genetico/` | `genetico.ipynb`, `genetico.py`, rede treinada, avaliações, curvas do treino e população final |
| `front_end/` | Interface do jogo |
| `relatorio/` | Relatório do T2 |
| `apoio/` | Regras, avaliação, treino pelo terminal, testes e dependências dos notebooks |

O comando para abrir o jogo continua sendo `python3 -m t2.app`, executado na raiz do repositório. Para a apresentação dos algoritmos, abra os três notebooks em `algoritmos/`.

Ao iniciar o servidor, a rede de `algoritmos/genetico/rede-t2.json` já fica carregada e o adversário selecionado é **Rede treinada**. Não precisa escolher arquivo para começar. Use **Carregar rede** apenas quando quiser testar outros pesos.
