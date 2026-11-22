import unittest
import subprocess
import sys
import json
from pathlib import Path
import numpy as np
from t2.algoritmos.genetico.genetico import Config, Trainer
from t2.algoritmos.mlp.mlp import Network
from t2.algoritmos.minimax.minimax import minimax, opponent_move
from t2.apoio.jogo import result
from t2.apoio.treinamento import evaluate, play
from t2.algoritmos.genetico.genetico import crossover, mutate
from t2.app import app

class CoreTests(unittest.TestCase):
    def test_minimax_never_loses_all_opponent_paths(self):
        def visit(board, turn, ai):
            end=result(board)
            if end is not None:
                self.assertNotEqual(end,-ai)
                return
            choices=minimax(tuple(board),turn)[1][:1] if turn==ai else [i for i,v in enumerate(board) if not v]
            for i in choices:
                child=board.copy();child[i]=turn
                visit(child,-turn,ai)
        visit([0]*9,1,1)
        visit([0]*9,1,-1)

    def test_network_forward_and_encoding(self):
        self.assertEqual(Network.size(9),180)
        w=np.zeros(180);w[-9:]=np.arange(9)
        n=Network(w)
        self.assertEqual(n.choose([0]*9,1)[0],8)
        self.assertEqual(n.choose([0]*8+[1],-1)[:2],(8,7))
        np.testing.assert_allclose(n.forward([1,-1,0,0,0,0,0,0,0],1),n.forward([-1,1,0,0,0,0,0,0,0],-1))

    def test_reward_and_trace_accounting(self):
        n=Network(np.zeros(180))
        for starts in [True,False]:
            p=play(n,'dificil',starts,42,True)
            self.assertAlmostEqual(sum(e['reward'] for e in p['events']),p['score'])
            self.assertGreater(p['invalid'],0)
            self.assertTrue(all(len([v for v in e['board'] if v])==i+1 for i,e in enumerate(p['events'])))

    def test_crossover_real_values_and_limits(self):
        a=np.full(180,-1.);b=np.full(180,1.)
        child=crossover(a,b,np.random.default_rng(2),.5)
        self.assertTrue(np.all(np.abs(child)<=1))
        self.assertTrue(np.any((child>-1)&(child<1)))

    def test_crossover_redraws_an_invalid_child(self):
        class Draws:
            def __init__(self):
                self.values = iter([1.25, .25])
                self.calls = 0
            def uniform(self, low, high):
                self.calls += 1
                return next(self.values)
        rng = Draws()
        a, b = np.array([-.8, .8]), np.array([.8, -.8])
        child = crossover(a,b,rng,.5)
        self.assertEqual(rng.calls,2)
        np.testing.assert_allclose(child,[-.4,.4])

    def test_mutation_redraws_without_clipping(self):
        class Draws:
            def __init__(self):self.values = iter([1.2,.9,-1.2,-.9])
            def random(self,size):return np.zeros(size)
            def normal(self,mean,sigma):return next(self.values)
        weights = np.array([.8,-.8])
        child = mutate(weights,Draws(),1.,.2)
        np.testing.assert_allclose(child,[.9,-.9])
        np.testing.assert_allclose(weights,[.8,-.8])

    def test_elitism_reproducibility_and_phase(self):
        c=Config(population=6,generations=5,medium_games=2,elite=2,patience=2)
        a,b=Trainer(c),Trainer(c)
        a.step();b.step()
        elites=a.population[:2].copy()
        a.step();b.step()
        self.assertTrue(all(any(np.array_equal(e,w) for w in a.population) for e in elites))
        np.testing.assert_allclose(a.population,b.population)
        self.assertEqual(a.history,b.history)
        while not a.stopped:a.step()
        self.assertIn(a.stopped,('Limite de gerações','Convergência: aptidão estável no conjunto de treino'))
        self.assertEqual(a.history[-1]['phase'],'medio+dificil')

    def test_evaluation_denominators(self):
        r=evaluate(Network(np.zeros(180)),4,True,measure=True)
        self.assertEqual(r['games'],6)
        self.assertEqual(r['vitoria']+r['empate']+r['derrota'],6)
        self.assertAlmostEqual(r['accuracy'],r['correct']/r['decisions'])

class APITests(unittest.TestCase):
    def setUp(self):self.client=app.test_client()
    def test_automatic_network_on_startup(self):
        root = Path(__file__).resolve().parents[3]
        code = """
import json
from t2.app import app
client = app.test_client()
print(json.dumps({
    'state': client.get('/api/state').json,
    'export': client.get('/api/export').json,
    'first': client.post('/api/move', json={'board': [0]*9, 'mode': 'rede'}).json,
    'second': client.post('/api/move', json={'board': [1]+[0]*8, 'mode': 'rede'}).json,
}))
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        saved = json.loads((root/'t2/algoritmos/genetico/rede-t2.json').read_text())
        self.assertEqual(data['export']['weights'], saved['weights'])
        self.assertEqual(data['state']['network'], {'hidden': saved['hidden'], 'parameters': len(saved['weights'])})
        self.assertIn(data['first']['move'], range(9))
        self.assertIn(data['second']['move'], range(1,9))

    def test_train_play_export_import(self):
        with self.client.get('/') as response:
            self.assertEqual(response.status_code,200)
        self.assertEqual(self.client.post('/api/init',json={'population':6,'generations':3,'medium_games':2}).status_code,200)
        self.assertEqual(self.client.post('/api/step',json={}).json['generation'],1)
        p=self.client.post('/api/move',json={'board':[0]*9,'mode':'rede'})
        self.assertEqual(p.status_code,200)
        self.assertIn(p.json['move'],range(9))
        r=self.client.post('/api/demo',json={'mode':'dificil','starts':False,'individual':0})
        self.assertTrue(r.json['events'])
        exported=self.client.get('/api/export').json
        self.assertEqual(self.client.post('/api/import',json=exported).status_code,200)
        self.assertEqual(self.client.post('/api/evaluate',json={}).status_code,200)
    def test_invalid_inputs(self):
        for config in [{'population':3},{'medium_games':3},{'mutation':2},{'elite':100}]:
            self.assertEqual(self.client.post('/api/init',json=config).status_code,400)
        for board in [[0]*8,[1]*9,[True]+[0]*8,[-1]+[0]*8]:
            self.assertEqual(self.client.post('/api/move',json={'board':board,'mode':'dificil'}).status_code,400)
        self.assertEqual(self.client.post('/api/import',json={'version':1,'hidden':9,'weights':[0]}).status_code,400)

if __name__=='__main__':unittest.main()
