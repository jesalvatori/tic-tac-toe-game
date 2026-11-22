# T2: Jogo da Velha com MLP, algoritmo genético e Minimax

## Executar no VS Code / Mac

Para abrir o projeto, o terminal precisa estar na pasta do repositório. Os comandos são estes:

```bash
git switch t2-rede-genetica
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-t2.txt
python3 app_t2.py
```

Depois é só acessar http://127.0.0.1:5002 no navegador. O T1 continua no `app.py`, com as dependências que já usava. No T2 a rede é outra: esse módulo não importa os classificadores, os datasets nem a MLP do T1.

Para jogar com a rede do experimento que já está no projeto, carregue `t2/resultados/rede-t2.json` pela interface. Também dá para começar uma população e acompanhar o treino, avançando uma geração de cada vez ou deixando executar automaticamente. Ao pedir para pausar, a geração que está rodando termina primeiro.

A rede fica na memória do servidor. Para guardar os pesos, é preciso exportar o JSON. Outra coisa: o servidor é local e as abas compartilham a mesma sessão de treinamento. Abrir outra aba não cria um treino separado.

## Rede neural

A topologia inicial tem 9 entradas, 9 neurônios na camada oculta e 9 saídas. Quando falamos nas duas camadas com pesos, estamos contando a oculta e a de saída. A entrada só recebe os dados do tabuleiro, sem fazer uma transformação.

Na entrada, a peça da própria rede vale 1, a do adversário vale -1 e uma casa vazia vale 0. Fazemos essa leitura pela perspectiva da rede para ela poder jogar tanto como X quanto como O, mantendo o mesmo significado dos valores.

A propagação fica assim:

- Camada oculta: `h = tanh(x @ W1 + b1)`.
- Camada de saída: `y = h @ W2 + b2`.

Depois olhamos qual saída tem o maior valor, e ela indica a casa escolhida. Esses valores são escores. Não estamos calculando a probabilidade de cada jogada.

O cromossomo guarda `[W1, b1, W2, b2]`, ou seja, os pesos e os bias das duas camadas. Para H neurônios ocultos, a conta é `9H + H + 9H + 9 = 19H + 9` parâmetros. Com H=9, são 180.

Os valores iniciais são sorteados de maneira uniforme em [-1,1]. O cruzamento e a mutação também mantêm os pesos nesse intervalo.

A parte importante aqui é quem altera esses valores: só o AG. O módulo T2 não usa gradiente, backpropagation, Adam, `fit` nem treino supervisionado. O NumPy entra para fazer as operações matriciais da rede. Ele faz as contas da propagação, mas o ajuste dos parâmetros fica com o algoritmo genético.

## Minimax e sequência de treinamento

O Minimax procura o melhor resultado possível, considerando as jogadas até chegar ao fim da partida. Nesta implementação, a avaliação é feita pelo ponto de vista do jogador da vez, na forma Negamax, equivalente ao Minimax. Usamos um cache para não recalcular estados que já foram avaliados.

Se houver mais de uma ação com o mesmo melhor valor, ele joga na de menor índice. Esse desempate é fixo, não tem sorteio.

No difícil, o adversário sempre chama o Minimax. Já no médio, a cada jogada ele sorteia entre usar Minimax (50%) e escolher uma casa livre aleatória (50%). Isso não significa que exatamente metade das jogadas de cada partida vai ser aleatória. É uma probabilidade aplicada em cada turno.

Os primeiros 40% das gerações usam várias partidas no médio. Depois, a segunda fase continua com essas mesmas partidas e acrescenta duas no difícil: uma com a rede começando e outra com o Minimax começando.

No médio usamos uma quantidade par de partidas para equilibrar quem começa. As sementes são as mesmas para todos os indivíduos e ficam fixas entre as gerações da mesma fase. Isso ajuda a comparar as aptidões sem mudar os sorteios da avaliação de uma rede para outra.

Quando muda a fase, reiniciamos o acompanhamento da estagnação. Agora há outro conjunto de partidas na avaliação, então não faria sentido continuar comparando a aptidão como se nada tivesse mudado.

## Aptidão e escolhas inválidas

A função usada foi:

`aptidão = (3*vitórias + empates - 3*derrotas - 2*escolhas_inválidas) / partidas`

A vitória recebe +3, o empate +1 e a derrota -3. Se a primeira escolha da rede for uma casa ocupada, ela recebe -2 naquele turno.

Mesmo com esse erro, a partida continua. O programa pega a próxima saída válida no ranking da própria rede. Então existem duas informações que precisamos olhar: qual foi a escolha original e qual jogada acabou sendo executada. A escolha ocupada fica no histórico e conta como erro tanto na validade quanto na acurácia.

Esse jeito de continuar a partida é uma ajuda do programa. Por isso, o resultado final da partida inclui essa ajuda e não quer dizer que a rede aprendeu 100% das regras sozinha. Ela pode conseguir um bom resultado e ainda ter tentado jogar numa posição ocupada no meio do caminho.

Para calcular a aptidão, entram somente os resultados das partidas e as escolhas inválidas. As jogadas ótimas do Minimax são usadas para medir acurácia nas avaliações e nas partidas contra a pessoa. Elas não entram na seleção dos indivíduos.

## Ciclo do AG

Começamos gerando N cromossomos com pesos reais. Cada um corresponde a uma rede, que joga suas partidas contra o adversário e recebe uma aptidão. Depois a população é ordenada da maior aptidão para a menor.

Os indivíduos de elite são copiados para a próxima geração sem alteração. Para escolher os pais dos outros indivíduos, usamos torneio sem reposição, com dois candidatos como padrão. Entre os candidatos sorteados, passa o que tem a maior aptidão.

O filho é gerado com BLX-alpha:

`filho_i = pai_i + beta_i*(mae_i-pai_i)`, com `beta_i ~ U(-alpha,1+alpha)`.

Na mutação, cada gene tem a probabilidade configurada de receber uma alteração. Essa alteração soma ruído normal `N(0,sigma)`, e o valor fica limitado a [-1,1]. O ciclo se repete até alcançar o limite de gerações ou a estagnação da melhor aptidão na segunda fase.

O BLX-alpha aparece na seção 3.2.2 do material `JAI-ag.pdf`, fornecido na disciplina. Como nossos genes são pesos contínuos, os operadores baseados em permutação não são adequados aqui. Para elitismo e torneio, seguimos as definições de `AlgoritmosGeneticos.pdf`.

A convergência usada para parar é a estabilidade no conjunto de treino. A rede parar de melhorar ali não significa, por si só, que ela já joga perfeitamente.

## Métricas

A acurácia de decisões é calculada como escolhas originais ótimas / decisões da rede. Para conferir isso, o Minimax fornece todas as ações que têm o valor máximo naquele estado. Se houver empate entre ações, qualquer uma delas pode contar como acerto. Escolher uma casa ocupada conta como erro.

Também medimos a taxa de escolhas válidas, que é escolhas originais livres / decisões. Ela mostra se a rede está respeitando as casas que já estão ocupadas, mesmo quando a escolha não é a melhor jogada possível.

As vitórias, os empates e as derrotas ficam separados. Taxa de vitórias e acurácia não são a mesma coisa neste trabalho.

As sementes do teste foram reservadas: no médio, a partir de 1.000.000, e no difícil, 2.000.000. Esse teste fica fora do AG, sem participar da seleção.

No difícil há duas partidas determinísticas, seguindo o desempate que definimos. Isso cobre essas trajetórias. Para afirmar que a rede se sai bem contra todas as sequências possíveis de jogadas ótimas, só essas partidas não bastam.

## Experimento incluído

O experimento do projeto usou 80 indivíduos, H=9, 250 gerações, 24 partidas no médio, 2 elites, torneio de 2, mutação 0,08, sigma 0,20, alpha 0,5 e semente 42. Nesse caso, o treinamento terminou pelo limite de gerações.

A comparação abaixo é entre a melhor rede da primeira geração e a melhor da última, usando o mesmo teste reservado. Vale lembrar o que estamos chamando de primeira geração: ela já escolhe a melhor entre 80 redes aleatórias. Portanto, não é a média do que uma rede aleatória qualquer faria.

| Medida no médio (100 partidas) | Primeira geração | Última geração |
| --- | ---: | ---: |
| Vitórias | 34 | 64 |
| Empates | 13 | 21 |
| Derrotas | 53 | 15 |
| Acurácia de decisões | 36,6% (122/333) | 77,9% (275/353) |
| Escolhas válidas | 50,8% | 82,4% |
| Escolhas ocupadas | 164 | 62 |

No difícil, a rede final empatou as duas partidas. Ela teve 8/9 decisões ótimas (88,9%) e uma escolha ocupada, que foi corrigida pela recuperação explicada antes.

O resultado melhorou ao longo desse experimento, mas a rede ainda perde no médio e ainda escolhe casas ocupadas. Como usamos uma única semente, não dá para assumir que outros sorteios de treinamento vão produzir o mesmo comportamento. Para comparar melhor, seria preciso variar as sementes e o tamanho da camada oculta e olhar os resultados dessas execuções também.

Se os parâmetros forem ajustados várias vezes olhando esse mesmo teste, ele acaba ajudando nessas escolhas. Nesse caso, depois não podemos apresentar o mesmo teste como uma avaliação independente. É algo para cuidar ao fazer novos experimentos.

Para repetir a execução:

```bash
python3 -m t2.train
python3 -m unittest discover -s tests -v
```

O `train` grava a rede e as métricas em `t2/resultados`. No JSON da rede também vai o histórico de cada geração, para consultar a evolução depois.

Os testes verificam o Minimax contra todos os caminhos do adversário. Também conferem a propagação, a codificação do tabuleiro, as penalidades, o BLX, o elitismo, a repetição dos resultados com a mesma configuração e a API.

## Demonstração e apresentação

Na demonstração, podemos começar mostrando as configurações e explicando de onde vêm os 180 parâmetros. Depois criamos uma população pequena e executamos uma geração, para conseguir olhar a aptidão e as escolhas inválidas dos indivíduos.

Com um indivíduo escolhido, avançamos a partida jogada por jogada. Nessa parte dá para mostrar a escolha original da rede, a ação que o programa executou e a penalidade, quando houver uma escolha inválida.

Depois carregamos a rede do experimento e jogamos contra ela, alternando quem começa. Por fim, executamos a avaliação reservada e comentamos os resultados, incluindo os erros que a rede ainda comete.

O PPT precisa explicar o objetivo, a topologia, a representação, a aptidão, os operadores, a sequência do treinamento, os resultados e as considerações finais. Cada integrante precisa entender a sua parte e aparecer no vídeo falando. A gravação e a apresentação em aula ficam com o grupo.

A professora indicou este vídeo: https://www.youtube.com/watch?v=P7XHzqZjXQs (IAMDinosaur). A reprodução e a transcrição não foram acessadas durante esta implementação. Ele fica como referência indicada, junto do enunciado e dos materiais do Moodle, sem substituir esses materiais.

## Organização dos arquivos

As pastas seguem a separação usada no T1: algoritmos, interface e relatório. Todos os arquivos desta parte ficam dentro de `t2/`, e o servidor continua sendo iniciado pelo `app_t2.py` na raiz do repositório.

| Arquivo ou pasta | O que fica ali |
| --- | --- |
| `t2/algoritmos/mlp/mlp.py` | A rede e a propagação das entradas |
| `t2/algoritmos/minimax/minimax.py` | O adversário nos modos médio e difícil |
| `t2/algoritmos/genetico/genetico.py` | Configuração, população e operadores do AG |
| `t2/jogo.py` | Regras para identificar vitória ou empate |
| `t2/treinamento.py` | Partidas, recompensas e avaliação das redes |
| `t2/front_end/` | HTML, CSS e JavaScript do jogo |
| `t2/resultados/` | Pesos da rede e resultados do experimento |
| `t2/relatorio/relatorio_t2.md` | Este relatório |
| `t2/train.py` | Execução do treino pelo terminal |

O T2 aprende jogando partidas contra o Minimax. Por isso não há uma pasta de dataset igual à do T1, que usava exemplos classificados em arquivos CSV.

## Executar os experimentos nos notebooks

Cada algoritmo tem seu notebook na mesma pasta do arquivo Python: `t2/algoritmos/mlp/mlp.ipynb`, `t2/algoritmos/minimax/minimax.ipynb` e `t2/algoritmos/genetico/genetico.ipynb`. A organização segue o T1. Primeiro instale as dependências com `python3 -m pip install -r requirements-notebook-t2.txt`. No VS Code, abra o notebook e escolha o kernel do ambiente `.venv`.

O notebook da MLP define a rede e mostra a propagação de um tabuleiro, além de consultar os pesos já treinados. O do Minimax define a busca, os modos e executa exemplos de escolha e de partida. O do genético define a configuração e os operadores e acompanha o treinamento completo, com a população, os gráficos, o teste e a exportação.

Cada notebook pode ser executado por conta própria, sem depender de variáveis de outra sessão. Os arquivos `.py` são os usados pela interface. Se alterarmos o código de um notebook, essa alteração não muda automaticamente o módulo Python correspondente.

Os valores iniciais do treinamento são os mesmos do experimento deste relatório. Se alterarmos a configuração e rodarmos outro treino, os resultados serão os daquela execução. Os arquivos do novo treino ficam em `t2/resultados/rede-notebook.json` e `t2/resultados/avaliacao-notebook.json`. Eles não substituem os arquivos que já acompanham o projeto. Para jogar com essa rede, abra a interface pelo Flask e use a opção de carregar rede.
