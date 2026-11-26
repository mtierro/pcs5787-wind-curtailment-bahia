"""Reajusta apenas candidatos com convergência insuficiente e preserva a tentativa."""
import json
import fcntl
from .common import ROOT,CACHE,OUT,atomic_json,atomic_frame,note
from .experiments import Campaign,select
import pandas as pd
import argparse


def converged(row):
    for ident in row['run_ids']:
        meta=json.loads((CACHE/f'{ident}.json').read_text())
        if any('failed to converge' in w.lower() for w in meta.get('warnings',[])):return False
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ablation',action='store_true');args=parser.parse_args()
    with (ROOT/'.scratch/v2/campaign.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.ablation:
            path=ROOT/'configs/v2/ablation_results.json';rows=json.loads(path.read_text())
            failed=[r for r in rows if not converged(r)]
            archive=ROOT/'configs/v2/ablation_original_results.json'
            if not archive.exists():atomic_json(archive,rows)
            specs=[{**r['spec'],'params':{**r['spec']['params'],'max_iter':3000}} for r in failed]
            atomic_json(ROOT/'configs/v2/ablation_numerical_grid.json',specs)
            repairs=select(specs,Campaign(),'ablation_numerical_results')
            replace={tuple(old['run_ids']):new for old,new in zip(failed,repairs)}
            atomic_json(path,[replace.get(tuple(r['run_ids']),r) for r in rows])
            note('R3 convergência das ablações',f'{len(failed)} ablações verificadas novamente com teto de 3.000 iterações; originals em ablation_original_results.json e ledger. Somente configurações convergidas sustentam interpretação.')
            return
        path=ROOT/'configs/v2/representation_results.json';rows=json.loads(path.read_text())
        failed=[r for r in rows if not converged(r)]
        specs=[]
        for r in failed:
            s=r['spec'];specs.append({**s,'family':'numerical_repair','params':{**s['params'],'max_iter':3000}})
        atomic_json(ROOT/'configs/v2/numerical_grid.json',specs)
        extra=select(specs,Campaign(),'numerical_results')
        unique={tuple(r['run_ids']):r for r in rows+extra}
        atomic_json(path,list(unique.values()))
        atomic_frame(pd.DataFrame([{'run_ids':'|'.join(r['run_ids']),'reason':r['spec']['reason'],'algorithm':r['spec']['algorithm'],'params':str(r['spec']['params']),'converged':converged(r),'mean_ap':r['mean_ap']} for r in failed+extra]),OUT/'tables/numerical_convergence.csv')
        note('R3 convergência',f'{len(failed)} configurações com limite de iterações reexaminadas com 3000 iterações. As tentativas originais permanecem no ledger; somente resultados convergidos são elegíveis à seleção posterior.')


if __name__=='__main__':main()
