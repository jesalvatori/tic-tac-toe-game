
const casas = document.querySelectorAll(".casa");
const mensagem = document.getElementById("mensagem");
const estado = document.getElementById("estado");
const previsao = document.getElementById("previsao");
const botaoReiniciar = document.getElementById("reiniciar");

const seletorAlgoritmo = document.getElementById("algoritmo");
const statusModelo = document.getElementById("status-modelo");

const campoVitoriasX = document.getElementById("vitorias-x");
const campoVitoriasO = document.getElementById("vitorias-o");
const campoEmpates = document.getElementById("empates");

const campoAcertos = document.getElementById("acertos-ia");
const campoErros = document.getElementById("erros-ia");
const campoAcuracia = document.getElementById("acuracia-ia");

let tabuleiro = Array(9).fill("");
let jogoFinalizado = false;
let vezComputador = false;
let aguardandoIA = false;
let erroServidor = false;
let temporizadorComputador = null;
let versaoPartida = 0;
let aviso = "";

let vitoriasX = 0;
let vitoriasO = 0;
let empates = 0;

// Contagens independentes para cada algoritmo
const metricas = {
    knn: { acertos: 0, erros: 0 },
    mlp: { acertos: 0, erros: 0 },
    arvore: { acertos: 0, erros: 0 },
    random_forest: { acertos: 0, erros: 0 },
    boosting: { acertos: 0, erros: 0 }
};

const combinacoesVitoria = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6]
];

// Estado real: calculado pelas regras do jogo
function verificarEstado() {
    for (const [a, b, c] of combinacoesVitoria) {
        if (
            tabuleiro[a] !== "" &&
            tabuleiro[a] === tabuleiro[b] &&
            tabuleiro[b] === tabuleiro[c]
        ) {
            return tabuleiro[a] === "X"
                ? "X venceu"
                : "O venceu";
        }
    }

    if (tabuleiro.every(casa => casa !== "")) {
        return "Empate";
    }

    return "Tem jogo";
}

function atualizarPlacar() {
    campoVitoriasX.textContent = vitoriasX;
    campoVitoriasO.textContent = vitoriasO;
    campoEmpates.textContent = empates;
}

function atualizarMetricas() {
    const algoritmo = seletorAlgoritmo.value;
    const dados = metricas[algoritmo];

    const total = dados.acertos + dados.erros;

    campoAcertos.textContent = dados.acertos;
    campoErros.textContent = dados.erros;

    campoAcuracia.textContent = total === 0
        ? "—"
        : `${(100 * dados.acertos / total).toFixed(1)}%`;
}

function modeloDisponivel() {
    return [
        "knn",
        "mlp",
        "arvore",
        "random_forest",
        "boosting"
    ].includes(seletorAlgoritmo.value);
}

function atualizarSelecao() {
    const nome = seletorAlgoritmo.options[
        seletorAlgoritmo.selectedIndex
    ].text;

    statusModelo.textContent = modeloDisponivel()
        ? `${nome} — integrado ao Python`
        : `${nome} — aguardando integração`;

    previsao.textContent = modeloDisponivel()
        ? "Aguardando jogada"
        : "Não disponível";

    atualizarMetricas();
    atualizarInterface();
}

function finalizarPartida(resultado) {
    if (jogoFinalizado) return;

    jogoFinalizado = true;

    if (resultado === "X venceu") {
        vitoriasX++;
        mensagem.textContent = "Parabéns! Você venceu!";
    } else if (resultado === "O venceu") {
        vitoriasO++;
        mensagem.textContent = "O computador venceu!";
    } else {
        empates++;
        mensagem.textContent = "A partida terminou empatada!";
    }

    atualizarPlacar();
}

function atualizarInterface() {
    estado.textContent = verificarEstado();

    casas.forEach((casa, indice) => {
        casa.textContent = tabuleiro[indice];

        casa.classList.toggle(
            "x",
            tabuleiro[indice] === "X"
        );

        casa.classList.toggle(
            "o",
            tabuleiro[indice] === "O"
        );

        casa.disabled =
            jogoFinalizado ||
            aguardandoIA ||
            vezComputador ||
            erroServidor ||
            !modeloDisponivel() ||
            tabuleiro[indice] !== "";
    });

    if (jogoFinalizado) return;

    if (!modeloDisponivel()) {
        mensagem.textContent =
            "Este algoritmo ainda não foi integrado.";
    } else if (erroServidor) {
        mensagem.textContent = aviso;
    } else if (aguardandoIA) {
        mensagem.textContent =
            "A IA está analisando o tabuleiro...";
    } else if (aviso) {
        mensagem.textContent = aviso;
    } else {
        mensagem.textContent = vezComputador
            ? "O computador está jogando..."
            : "Sua vez! Escolha uma casa.";
    }
}

// Consulta o classificador Python após cada jogada
async function avaliarJogada(versao) {
    aguardandoIA = true;
    atualizarInterface();

    try {
        const resposta = await fetch("/api/prever", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                algoritmo: seletorAlgoritmo.value,
                tabuleiro: [...tabuleiro]
            })
        });

        const dados = await resposta.json();

        if (!resposta.ok) {
            throw new Error(
                dados.erro || "Falha na classificação."
            );
        }

        // Ignorar respostas de partidas já reiniciadas
        if (versao !== versaoPartida) {
            return false;
        }

        const resultadoReal = verificarEstado();
        const resultadoIA = dados.previsao;

        previsao.textContent = resultadoIA;

        const estatisticas = metricas[
            seletorAlgoritmo.value
        ];

        if (resultadoIA === resultadoReal) {
            estatisticas.acertos++;
        } else {
            estatisticas.erros++;
        }

        atualizarMetricas();

        aguardandoIA = false;
        aviso = "";

        // Um fim real encerra a partida,
        // mesmo quando a IA não o reconhece.
        if (resultadoReal !== "Tem jogo") {
            finalizarPartida(resultadoReal);
        } else if (resultadoIA !== "Tem jogo") {
            // Falso positivo: registrar erro e continuar
            aviso =
                "A IA previu o fim, mas ainda tem jogo!";
        }

        atualizarInterface();

        return !jogoFinalizado;

    } catch (erro) {
        if (versao !== versaoPartida) {
            return false;
        }

        aguardandoIA = false;
        erroServidor = true;

        aviso = `Erro ao consultar a IA: ${erro.message}`;

        atualizarInterface();
        return false;
    }
}

async function jogarHumano(posicao) {
    if (
        jogoFinalizado ||
        vezComputador ||
        aguardandoIA ||
        erroServidor ||
        !modeloDisponivel() ||
        tabuleiro[posicao] !== ""
    ) {
        return;
    }

    tabuleiro[posicao] = "X";
    vezComputador = true;
    aviso = "";

    const versao = versaoPartida;
    const continuar = await avaliarJogada(versao);

    if (!continuar) return;

    temporizadorComputador = setTimeout(
        () => jogarComputador(versao),
        500
    );

    atualizarInterface();
}

async function jogarComputador(versao) {
    temporizadorComputador = null;

    if (
        versao !== versaoPartida ||
        jogoFinalizado ||
        erroServidor
    ) {
        return;
    }

    const disponiveis = [];

    tabuleiro.forEach((valor, indice) => {
        if (valor === "") {
            disponiveis.push(indice);
        }
    });

    if (disponiveis.length === 0) return;

    const sorteio = Math.floor(
        Math.random() * disponiveis.length
    );

    const posicao = disponiveis[sorteio];

    tabuleiro[posicao] = "O";
    vezComputador = false;
    aviso = "";

    await avaliarJogada(versao);
}

function reiniciarJogo() {
    versaoPartida++;

    if (temporizadorComputador !== null) {
        clearTimeout(temporizadorComputador);
        temporizadorComputador = null;
    }

    tabuleiro = Array(9).fill("");
    jogoFinalizado = false;
    vezComputador = false;
    aguardandoIA = false;
    erroServidor = false;
    aviso = "";

    previsao.textContent = modeloDisponivel()
        ? "Aguardando jogada"
        : "Não disponível";

    atualizarInterface();
}

casas.forEach(casa => {
    casa.addEventListener("click", () => {
        jogarHumano(Number(casa.dataset.posicao));
    });
});

botaoReiniciar.addEventListener(
    "click",
    reiniciarJogo
);

seletorAlgoritmo.addEventListener("change", () => {
    reiniciarJogo();
    atualizarSelecao();
});

atualizarSelecao();
atualizarPlacar();
