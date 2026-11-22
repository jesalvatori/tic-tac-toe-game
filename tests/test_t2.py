import unittest
import numpy as np
from t2.core import Config, Network, Trainer, crossover, evaluate, minimax, play, result
from app_t2 import app

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
