"""Servidor independente do T2. Execute: python app_t2.py."""
from pathlib import Path
from threading import Lock
import json
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from t2.core import Config, Trainer, Network, evaluate, play, opponent_move, minimax, result

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT/'t2/web'), static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 1024*1024
lock = Lock()
trainer = None
loaded = None

@app.errorhandler(ValueError)
def bad_input(error):
    return jsonify(error=str(error)), 400

@app.errorhandler(TypeError)
def bad_type(error):
    return jsonify(error='Dados incompatíveis com a operação.'), 400

def model():
    net = trainer.best if trainer and trainer.best else loaded
    if net is None:
        raise ValueError('Treine pelo menos uma geração ou carregue uma rede.')
    return net

@app.get('/')
def index():
    return send_from_directory(app.static_folder,'index.html')

@app.post('/api/init')
def initialize():
    global trainer, loaded
    payload = request.get_json()
    if not isinstance(payload,dict):
        raise ValueError('Configuração deve ser um objeto JSON.')
    with lock:
        trainer = Trainer(Config(**payload))
        loaded = None
        return jsonify(trainer.state())

@app.post('/api/step')
def step():
    with lock:
        if trainer is None:
            raise ValueError('Inicialize a população primeiro.')
        return jsonify(trainer.step())

@app.get('/api/state')
def state():
    with lock:
        return jsonify(trainer.state() if trainer else {})

@app.post('/api/evaluate')
def test():
    with lock:
        net = model()
        # Sementes reservadas, não usadas na seleção do AG.
        return jsonify(medium=evaluate(net,100,False,1000000,True),
                       hard=evaluate(net,0,True,2000000,True))

@app.post('/api/demo')
def demo():
    p = request.get_json()
    mode = p.get('mode','dificil')
    if mode not in ('medio','dificil') or type(p.get('starts',True)) is not bool:
        raise ValueError('Modo ou jogador inicial inválido.')
    with lock:
        net = model()
        if 'individual' in p:
            i = p['individual']
            if trainer is None or not trainer.ranked or type(i) is not int or not 0 <= i < len(trainer.population):
                raise ValueError('Indivíduo inexistente.')
            net = Network(trainer.population[i],trainer.config.hidden)
        return jsonify(play(net,mode,p.get('starts',True),1000000,True,True))

def validated_board(p):
    board = p.get('board')
    if not isinstance(board,list) or len(board) != 9 or any(type(v) is not int or v not in (-1,0,1) for v in board):
        raise ValueError('Tabuleiro inválido.')
    x,o = board.count(1),board.count(-1)
    if x not in (o,o+1) or result(board) is not None:
        raise ValueError('Partida encerrada ou turnos inválidos.')
    return board, 1 if x == o else -1

@app.post('/api/move')
def move():
    p = request.get_json()
    board, player = validated_board(p)
    mode = p.get('mode')
    if mode == 'rede':
        with lock:
            net = model()
            raw, pos, outputs = net.choose(board,player)
        optimal = minimax(tuple(board),player)[1]
        return jsonify(move=pos, raw=raw, outputs=outputs, invalid=board[raw]!=0, correct=raw in optimal)
    if mode not in ('medio','dificil'):
        raise ValueError('Adversário desconhecido.')
    return jsonify(move=opponent_move(board,player,mode,np.random.default_rng()))

@app.get('/api/export')
def export():
    with lock:
        net = model()
        payload = dict(version=1,hidden=net.hidden,weights=net.weights.tolist(),
                       config=trainer.state()['config'] if trainer else None,
                       history=trainer.history if trainer else [])
    return app.response_class(json.dumps(payload,ensure_ascii=False),mimetype='application/json',
                              headers={'Content-Disposition':'attachment; filename=rede-t2.json'})

@app.post('/api/import')
def import_model():
    global loaded, trainer
    p = request.get_json()
    if not isinstance(p,dict) or p.get('version') != 1 or type(p.get('hidden')) is not int or not 1 <= p['hidden'] <= 64:
        raise ValueError('Arquivo de rede inválido.')
    net = Network(p.get('weights'),p['hidden'])
    if np.any(np.abs(net.weights)>1):
        raise ValueError('Pesos devem estar entre -1 e 1.')
    with lock:
        loaded, trainer = net, None
    return jsonify(parameters=len(net.weights),hidden=net.hidden)

if __name__ == '__main__':
    print('T2: http://127.0.0.1:5002')
    app.run(host='127.0.0.1',port=5002,debug=False)
