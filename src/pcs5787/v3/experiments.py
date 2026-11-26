"""Correção OOF e seleção interna cronológica; externo bloqueado por padrão."""
import json,time,resource
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from ..v2.experiments import metrics
from ..v2.learners import make_model
from .common import ROOT,DATA,OUT,CACHE,MODELS,P,atomic_json,atomic_frame,digest,sha256,stamp,cutoff,population,training_mask,note,frame_digest
from .engine import Engine
from .models import OffsetModel,safe_logit

def initial_candidates():
    common=dict(kind='tree',calendar=True,penalty=40,leaves=7,iterations=120,months=6,source_A='A_tree')
    cases=[('calendar',[]),('missing',['missing']),('weather_v2',['v2']),('aggregate_now',['aggregate_now','missing']),('sensor_now',['sensor_now','missing']),('aggregate_trajectory',['aggregate_trajectory','missing']),('sensor_trajectory',['sensor_trajectory','missing']),('coherence',['aggregate_trajectory','coherence','missing'])]
    configs=[dict(common,name=n,groups=g) for n,g in cases]
    configs += [dict(common,name='weather_only',groups=['v2'],calendar=False),dict(common,name='linear_weather',groups=['v2'],kind='linear',penalty=400),dict(common,name='linear_trajectory',groups=['sensor_trajectory','coherence','missing'],kind='linear',penalty=400)]
    return configs

class CorrectionCampaign:
    def __init__(self):
        self.e=Engine();self.oof=pd.read_parquet(DATA/'base_oof.parquet')
        self.code={p.name:sha256(p) for p in [ROOT/'src/pcs5787/v3/models.py',ROOT/'src/pcs5787/v3/experiments.py']}
        self.ledger_path=OUT/'tables/correction_ledger.csv';self.ledger=pd.read_csv(self.ledger_path).to_dict('records') if self.ledger_path.exists() else []

    def run(self,config,origin,end,seed=5787,allow_final=False):
        if not allow_final:assert stamp(end)<=stamp('2026-04-01')
        source=config.get('source_A','A_tree');history=self.oof[self.oof.name.eq(source)&self.oof.time_local.lt(cutoff(origin))&self.oof.time_local.ge(cutoff(origin)-pd.DateOffset(months=config['months']))].copy()
        identity=dict(config=config,origin=origin,end=end,seed=seed,data=self.e.dataset,oof_rows=frame_digest(history),code=self.code,feature_groups=sha256(ROOT/'configs/v3/feature_groups.json'),dependencies=sha256(ROOT/'configs/v3/requirements.lock.txt'))
        ident=digest(identity)[:24];path=CACHE/f'correction_{ident}.parquet';meta_path=CACHE/f'correction_{ident}.json'
        if path.exists() and meta_path.exists():
            meta=json.loads(meta_path.read_text());assert sha256(path)==meta['pred_sha256'];return pd.read_parquet(path),meta
        start=time.monotonic()
        assert history.time_local.max()+pd.Timedelta(hours=169)<=stamp(origin)-pd.Timedelta(hours=1)
        # The A score for every correction-training row was predicted before its outcome was available.
        assert (pd.to_datetime(history.train_last)+pd.Timedelta(hours=169)<=history.time_local-pd.Timedelta(hours=1)).all()
        train=history.merge(self.e.base,on=['region','time_local'],validate='one_to_one')
        valid_p=self.e.predict_base(source,origin,end,seed=seed,allow_final=allow_final)
        valid=valid_p.merge(self.e.base,on=['region','time_local'],validate='one_to_one')
        train['base_logit']=safe_logit(train.p);valid['base_logit']=safe_logit(valid.p)
        columns=list(dict.fromkeys(['base_logit']+sum([self.e.groups[k] for k in config['groups']],[])))
        if config.get('exclude'):columns=[c for c in columns if not any(token in c for token in config['exclude'])]
        model=OffsetModel(columns,config['kind'],config['calendar'],config['penalty'],config['leaves'],config['iterations'],seed)
        with threadpool_limits(limits=4):model.fit(train,train.y,train.base_logit);g=model.correction(valid)
        output=valid_p.copy();output['p_A']=output.p;output['g']=g
        output['name']=config['name'];output['run_id']=ident;output['seed']=seed
        atomic_frame(output,path);model_path=MODELS/f'correction_{ident}.joblib';joblib.dump(model,model_path,compress=3)
        meta=dict(id=ident,identity=identity,n_train=len(train),n_eval=len(valid),train_first=str(train.time_local.min()),train_last=str(train.time_local.max()),columns=columns,seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,pred_sha256=sha256(path),model_sha256=sha256(model_path),converged=getattr(model,'converged',True))
        atomic_json(meta_path,meta);atomic_json(model_path.with_suffix('.json'),meta)
        self.ledger=[r for r in self.ledger if r['id']!=ident]+[dict(id=ident,name=config['name'],origin=origin,end=end,seed=seed,n_train=len(train),n_eval=len(valid),train_last=meta['train_last'],seconds=meta['seconds'],peak_rss_mib=meta['peak_rss_mib'],converged=meta['converged'],status='done')]
        atomic_frame(pd.DataFrame(self.ledger),self.ledger_path)
        print('Correction',config['name'],origin,'->',end,len(train),f'{meta["seconds"]:.2f}s',flush=True)
        return output,meta

    def development(self,configs,stage='cycle1'):
        allmetrics=[];selected=[];outer=[];allpred=[]
        for f in P['development_folds']:
            for config in configs:
                inner=[]
                for origin in f['inner_origins']:
                    end=min(stamp(origin)+pd.DateOffset(months=3),cutoff(f['origin']))
                    d,_=self.run(config,origin,str(end.tz_localize(None)))
                    for strength in P['lambda_candidates']:
                        q=d.p_A.to_numpy() if strength==0 else expit(safe_logit(d.p_A)+strength*d.g)
                        row=dict(stage=stage,fold=f['name'],role='inner',name=config['name'],origin=origin,strength=strength,**metrics(d.y,q));allmetrics.append(row);inner.append(row)
                mean=pd.DataFrame(inner).groupby('strength')[['ap','brier','log_loss']].mean()
                # Each scoring question chooses lambda from past inner predictions, independently of the outer fold.
                choices={'log_loss':float(mean.log_loss.idxmin()),'ap':float(mean.ap.idxmax())}
                d,meta=self.run(config,f['origin'],f['end'])
                allpred.append(d.assign(fold=f['name'],stage=stage))
                for objective,strength in choices.items():
                    q=d.p_A.to_numpy() if strength==0 else expit(safe_logit(d.p_A)+strength*d.g)
                    row=dict(stage=stage,fold=f['name'],role='outer',name=config['name'],origin=f['origin'],objective=objective,strength=strength,**metrics(d.y,q));allmetrics.append(row)
                    z=d.copy();z['p']=q;z['strength']=strength;z['fold']=f['name'];z['objective']=objective;z['stage']=stage;outer.append(z)
                    selected.append(dict(stage=stage,fold=f['name'],name=config['name'],objective=objective,strength=strength,inner_mean=mean.reset_index().to_dict('records'),config=config,run_id=meta['id']))
        result=pd.DataFrame(allmetrics)
        atomic_frame(result,OUT/f'tables/{stage}_metrics.csv')
        atomic_frame(pd.concat(outer,ignore_index=True),DATA/f'{stage}_selected_predictions.parquet')
        atomic_frame(pd.concat(allpred,ignore_index=True),DATA/f'{stage}_outer_components.parquet')
        atomic_json(ROOT/f'configs/v3/{stage}_selection.json',selected)
        means=result[result.role.eq('outer')].groupby(['name','objective'])[['ap','brier','log_loss','f1','recall']].mean().sort_values('ap',ascending=False)
        atomic_frame(means.reset_index(),OUT/f'tables/{stage}_mean_metrics.csv');print(means.to_string(),flush=True)
        note(stage,f'{len(configs)} correções × três folds, intensidade escolhida em duas origens internas anteriores para AP e log loss. Previsões/seleção salvas. Externo ainda bloqueado; inspecionar estabilidade e controles antes de refinar.')

def main():
    configs=initial_candidates();atomic_json(ROOT/'configs/v3/cycle1_candidates.json',configs)
    CorrectionCampaign().development(configs)

if __name__=='__main__':main()
