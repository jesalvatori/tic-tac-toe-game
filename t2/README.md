# T2 — Jogo da Velha

A documentação completa está em [relatorio/relatorio_t2.md](relatorio/relatorio_t2.md), junto da parte do relatório, como na organização do T1.

Para abrir o jogo, execute na raiz do repositório:

```bash
python3 -m t2.app
```

Acesse http://127.0.0.1:5002. Os algoritmos ficam em `t2/algoritmos/`, a interface em `t2/front_end/` e os arquivos do experimento em `t2/resultados/`.

Os notebooks ficam junto de cada algoritmo:

- [MLP](algoritmos/mlp/mlp.ipynb)
- [Minimax](algoritmos/minimax/minimax.ipynb)
- [Algoritmo genético](algoritmos/genetico/genetico.ipynb)

Abra no VS Code e selecione o kernel da `.venv`. As dependências estão em `t2/requirements-notebook.txt`.

Cada notebook tem as explicações, o código e as saídas do próprio algoritmo. Os gráficos já estão salvos no arquivo; para refazer os experimentos, execute as células na ordem com **Run All**.

| Notebook | O que observar |
| --- | --- |
| MLP | Escores das nove casas, mapas dos pesos, distribuição dos parâmetros e comparação entre uma rede aleatória e a rede treinada. |
| Minimax | Valor de cada jogada disponível, frequência das escolhas no médio e no difícil e partidas contra um jogador aleatório. |
| Genético | Aptidão e jogadas válidas ao longo das gerações, resultados das partidas, diversidade da população, aptidões finais e comparação entre a primeira e a última geração. |

No genético, a linha tracejada marca a entrada do modo difícil no treino. A avaliação passa a incluir mais duas partidas nessa fase, então os resultados ao longo das gerações aparecem em percentuais. A diversidade é a média dos desvios padrão de cada gene na população; ela ajuda a ver se os indivíduos estão ficando parecidos.

As comparações extras da MLP e do Minimax são demonstrações dos notebooks. Os resultados do experimento original continuam em `resultados/rede-t2.json` e `resultados/avaliacao.json`. Executar o notebook genético salva novos arquivos com o sufixo `-notebook`, sem sobrescrever esses dois.
