"""
Executa o treinamento pelo terminal, sem precisar abrir a interface.

Com python -m t2.train conseguimos repetir o treino com a configuração
definida e guardar a rede junto dos resultados.
"""
import argparse
import json
from pathlib import Path
from dataclasses import asdict
from t2.core import Config, Trainer, evaluate

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generations',type=int,default=250)
    parser.add_argument('--population',type=int,default=80)
    parser.add_argument('--hidden',type=int,default=9)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',default='t2/results')
    args=parser.parse_args()
    trainer=Trainer(Config(generations=args.generations,population=args.population,hidden=args.hidden,seed=args.seed))
    baseline=None
    while not trainer.stopped:
        trainer.step()
        if baseline is None:
            baseline=dict(medium=evaluate(trainer.best,100,False,1000000,True),hard=evaluate(trainer.best,0,True,2000000,True))
        if trainer.generation%25==0:
            h=trainer.history[-1]
            print(f"G {trainer.generation}: {h['best']:.3f}, válidas {h['valid_rate']:.1%}",flush=True)
    final=dict(medium=evaluate(trainer.best,100,False,1000000,True),hard=evaluate(trainer.best,0,True,2000000,True))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'rede-t2.json').write_text(json.dumps(dict(version=1,hidden=trainer.best.hidden,weights=trainer.best.weights.tolist(),config=asdict(trainer.config),history=trainer.history),ensure_ascii=False,indent=2))
    report=dict(config=asdict(trainer.config),generations=trainer.generation,stop=trainer.stopped,baseline=baseline,final=final)
    (out/'avaliacao.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
