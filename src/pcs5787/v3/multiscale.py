"""Duas memórias de calendário: referência forte e substituição meteorológica pareada."""
import numpy as np
import pandas as pd
from scipy.special import expit
from ..v2.experiments import metrics
from .common import ROOT,DATA,OUT,P,atomic_frame,atomic_json,stamp,cutoff,note
from .engine import Engine
from .models import safe_logit
from .refinement import JointCampaign
from .experiments import CorrectionCampaign

def logistic_candidates():
    base=dict(kind='linear',calendar=False,penalty=1000,leaves=3,iterations=100,source_A='A_log')
    rows=[dict(base,name=f'log_ridge_{name}_{months}',groups=groups,months=months) for months in [3,6] for name,groups in [('score',[]),('weather',['v2']),('trajectory',['sensor_trajectory','coherence','missing'])]]
    rows += [dict(base,name='log_tree_'+name,groups=groups,months=3,kind='tree',penalty=200) for name,groups in [('score',[]),('weather',['v2']),('trajectory',['sensor_trajectory','coherence','missing'])]]
    return rows

def blend(d,weight):
    if weight==0:return d.p_tree.to_numpy()
    if weight==1:return d.p_log.to_numpy()
    return expit((1-weight)*safe_logit(d.p_tree)+weight*safe_logit(d.p_log))

class MultiscaleCampaign:
    def __init__(self):self.j=JointCampaign();self.e=self.j.e

    def components(self,config,origin,end,seed=5787,allow_final=False):
        if config['branch']=='joint':
            d,_=self.j.run(config['joint'],origin,end,seed,allow_final);d['p_tree']=expit(safe_logit(d.p_A)+d.g)
        else:
            d=self.e.predict_base(config['branch'],origin,end,seed=seed,allow_final=allow_final);d['p_tree']=d.p
        log=self.e.predict_base('A_log',origin,end,seed=seed,allow_final=allow_final)
        d=d.merge(log[['region','time_local','p']].rename(columns={'p':'p_log'}),on=['region','time_local'],validate='one_to_one');d['name']=config['name'];return d

    def development(self,configs):
        rows=[];pred=[];selected=[]
        for f in P['development_folds']:
            for c in configs:
                inner=[]
                for origin in f['inner_origins']:
                    end=str(min(stamp(origin)+pd.DateOffset(months=3),cutoff(f['origin'])).tz_localize(None))
                    d=self.components(c,origin,end)
                    for weight in [0,.25,.5,.75,1]:inner.append(dict(weight=weight,**metrics(d.y,blend(d,weight))))
                mean=pd.DataFrame(inner).groupby('weight')[['ap','log_loss']].mean();d=self.components(c,f['origin'],f['end'])
                for objective,weight in [('ap',float(mean.ap.idxmax())),('log_loss',float(mean.log_loss.idxmin()))]:
                    z=d.copy();z['p']=blend(d,weight);z['weight_log']=weight;z['objective']=objective;z['fold']=f['name'];pred.append(z)
                    rows.append(dict(name=c['name'],fold=f['name'],objective=objective,weight_log=weight,**metrics(z.y,z.p)))
                    selected.append(dict(config=c,fold=f['name'],objective=objective,weight_log=weight,inner_mean=mean.reset_index().to_dict('records')))
        atomic_frame(pd.concat(pred,ignore_index=True),DATA/'multiscale_selected_predictions.parquet');atomic_frame(pd.DataFrame(rows),OUT/'tables/multiscale_metrics.csv');atomic_json(ROOT/'configs/v3/multiscale_selection.json',selected)
        mean=pd.DataFrame(rows).groupby(['name','objective'])[['ap','brier','log_loss']].mean().sort_values('ap',ascending=False);atomic_frame(mean.reset_index(),OUT/'tables/multiscale_mean_metrics.csv');print(mean.to_string(),flush=True)

def main():
    note('H8/H13 diagnóstico de referência','A logística AP média .71649 supera árvore .70002, B V2 .70801 e trajetórias .70733; F1 favorece logística, F2 árvore. H13: combinar duas memórias de calendário com peso escolhido apenas nas duas origens internas; substituir ramo árvore por B mantém ramo logístico igual. Controle A multiescala obrigatório. Correções sobre A logística usam OOF do próprio gerador. Nenhum ganho sobre árvore será chamado ganho sobre melhor A.')
    rows=logistic_candidates();atomic_json(ROOT/'configs/v3/logistic_correction_candidates.json',rows);CorrectionCampaign().development(rows,'logistic_correction')
    configs=[dict(name='multi_A',branch='A_tree'),dict(name='multi_B_v2',branch='B_v2')]
    for name,groups in [('trajectory',['sensor_trajectory','missing']),('coherence',['aggregate_trajectory','coherence','missing'])]:
        configs.append(dict(name='multi_'+name,branch='joint',joint=dict(name='joint_'+name if name=='coherence' else 'joint_sensor_trajectory',groups=groups,penalty=40,months=3,kind='joint',source_A='A_tree')))
    atomic_json(ROOT/'configs/v3/multiscale_candidates.json',configs);MultiscaleCampaign().development(configs)
    note('H13 execução concluída','Misturas com seleção anterior por AP/log loss e correções do gerador logístico salvas. Avaliar estabilidade contra todas as referências, sem escolher referência por resultado externo.')

if __name__=='__main__':main()
