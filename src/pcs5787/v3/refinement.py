"""Ciclo 2: separar calibração de informação meteorológica e testar B conjunto."""
import json,time,resource
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from ..v2.learners import make_model
from .common import ROOT,DATA,OUT,CACHE,MODELS,atomic_json,atomic_frame,digest,sha256,stamp,cutoff,population,training_mask,note
from .models import safe_logit,OffsetModel
from .experiments import CorrectionCampaign

def candidates():
    common=dict(kind='linear',calendar=False,penalty=1000,leaves=3,iterations=100,source_A='A_tree')
    rows=[]
    for months in [3,6,12]:
        for name,groups in [('score',[]),('weather',['v2']),('trajectory',['sensor_trajectory','coherence','missing'])]:
            rows.append(dict(common,name=f'ridge_{name}_{months}',groups=groups,months=months))
    for name,groups in [('score',[]),('missing',['missing']),('weather',['v2']),('trajectory',['sensor_trajectory','coherence','missing'])]:
        rows.append(dict(common,name='small_tree_'+name,groups=groups,months=3,kind='tree',penalty=200))
    return rows

class JointCampaign(CorrectionCampaign):
    def run(self,config,origin,end,seed=5787,allow_final=False):
        if not allow_final:assert stamp(end)<=stamp('2026-04-01')
        cols=list(dict.fromkeys(sum([self.e.groups[k] for k in config['groups']],[])))
        if config.get('exclude'):cols=[c for c in cols if not any(token in c for token in config['exclude'])]
        identity=dict(config=config,columns=cols,origin=origin,end=end,seed=seed,data=self.e.dataset,code=sha256(ROOT/'src/pcs5787/v3/refinement.py'))
        ident=digest(identity)[:24];path=CACHE/f'joint_{ident}.parquet';meta_path=CACHE/f'joint_{ident}.json'
        if path.exists() and meta_path.exists():return pd.read_parquet(path),json.loads(meta_path.read_text())
        reason=config.get('reason','ENE')
        start=time.monotonic();mask,target=population(self.e.base,reason,'spatial_recent');tr=self.e.base[mask&training_mask(self.e.base,origin,config.get('strategy','window3'))].copy()
        valid_p=self.e.predict_base('A_tree',origin,end,reason=reason,seed=seed,allow_final=allow_final);va=valid_p.merge(self.e.base,on=['region','time_local'],validate='one_to_one')
        if config.get('shift_hours'):
            # Negative control: shift only within each train/evaluation partition,
            # on an hourly grid, never wrap future measurements into earlier rows.
            for frame in [tr,va]:
                shifted=frame[['region','time_local']+cols].copy();shifted['time_local']+=pd.Timedelta(hours=config['shift_hours'])
                shifted=frame[['region','time_local']].merge(shifted,on=['region','time_local'],how='left',validate='one_to_one')
                frame[cols]=shifted[cols].to_numpy()
        spec=self.e.spec('A_tree',reason);spec['seed']=seed;spec['params']=dict(spec['params'],l2=config['penalty'])
        if config.get('leaves'):spec['params']['leaves']=config['leaves']
        model=make_model(spec,cols)
        with threadpool_limits(limits=4):model.fit(tr,tr[target],model__categorical_feature=list(range(6)));p=model.predict_proba(va)[:,1]
        out=valid_p.copy();out['p_A']=out.p;out['g']=safe_logit(p)-safe_logit(out.p_A);out['name']=config['name'];out['run_id']=ident;out['seed']=seed
        atomic_frame(out,path);mp=MODELS/f'joint_{ident}.joblib';joblib.dump(model,mp,compress=3)
        meta=dict(id=ident,identity=identity,n_train=len(tr),n_eval=len(va),train_first=str(tr.time_local.min()),train_last=str(tr.time_local.max()),seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,pred_sha256=sha256(path),model_sha256=sha256(mp),converged=True)
        atomic_json(meta_path,meta);atomic_json(mp.with_suffix('.json'),meta)
        self.ledger=[r for r in self.ledger if r['id']!=ident]+[dict(id=ident,name=config['name'],origin=origin,end=end,seed=seed,n_train=len(tr),n_eval=len(va),train_last=meta['train_last'],seconds=meta['seconds'],peak_rss_mib=meta['peak_rss_mib'],converged=True,status='done')];atomic_frame(pd.DataFrame(self.ledger),self.ledger_path)
        print('Joint',config['name'],origin,f'{meta["seconds"]:.2f}s',flush=True);return out,meta

class CompatibleCampaign(CorrectionCampaign):
    def run(self,config,origin,end,seed=5787,allow_final=False):
        if not allow_final:assert stamp(end)<=stamp('2026-04-01')
        start=time.monotonic();fit_origin=str((stamp(origin)-pd.DateOffset(months=3)).date())
        identity=dict(config=config,origin=origin,end=end,fit_origin=fit_origin,seed=seed,data=self.e.dataset,code={p.name:sha256(p) for p in [ROOT/'src/pcs5787/v3/refinement.py',ROOT/'src/pcs5787/v3/models.py']})
        ident=digest(identity)[:24];path=CACHE/f'compatible_{ident}.parquet';meta_path=CACHE/f'compatible_{ident}.json'
        if path.exists() and meta_path.exists():return pd.read_parquet(path),json.loads(meta_path.read_text())
        # A generator is fixed before all correction-training outcomes and remains fixed at evaluation.
        all_p=self.e.predict_base('A_tree',fit_origin,end,seed=seed,allow_final=allow_final)
        hist=all_p[all_p.time_local.lt(cutoff(origin))];vp=all_p[all_p.time_local.ge(stamp(origin))]
        tr=hist.merge(self.e.base,on=['region','time_local'],validate='one_to_one');va=vp.merge(self.e.base,on=['region','time_local'],validate='one_to_one')
        tr['base_logit']=safe_logit(tr.p);va['base_logit']=safe_logit(va.p)
        cols=list(dict.fromkeys(['base_logit']+sum([self.e.groups[k] for k in config['groups']],[])))
        model=OffsetModel(cols,config['kind'],config['calendar'],config['penalty'],config['leaves'],config['iterations'],seed)
        with threadpool_limits(limits=4):model.fit(tr,tr.y,tr.base_logit);g=model.correction(va)
        out=vp.copy();out['p_A']=out.p;out['g']=g;out['name']=config['name'];out['run_id']=ident;out['seed']=seed
        atomic_frame(out,path);mp=MODELS/f'compatible_{ident}.joblib';joblib.dump(model,mp,compress=3)
        meta=dict(id=ident,identity=identity,n_train=len(tr),n_eval=len(va),train_first=str(tr.time_local.min()),train_last=str(tr.time_local.max()),seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,pred_sha256=sha256(path),model_sha256=sha256(mp),converged=getattr(model,'converged',True),generator_same_before_after_correction=True)
        atomic_json(meta_path,meta);atomic_json(mp.with_suffix('.json'),meta)
        self.ledger=[r for r in self.ledger if r['id']!=ident]+[dict(id=ident,name=config['name'],origin=origin,end=end,seed=seed,n_train=len(tr),n_eval=len(va),train_last=meta['train_last'],seconds=meta['seconds'],peak_rss_mib=meta['peak_rss_mib'],converged=meta['converged'],status='done')];atomic_frame(pd.DataFrame(self.ledger),self.ledger_path)
        print('Compatible',config['name'],origin,f'{meta["seconds"]:.2f}s',flush=True);return out,meta

def main():
    note('C4 refinamento motivado','Ciclo 1: A árvore AP média ~.7000; nenhuma correção superou essa referência em AP média, embora weather_only reduza log loss para .3865 versus ~.4281. Corrigir só calendário não explica essa melhora, mas falta controle escalar do próprio score. Ciclo 2 adiciona esse controle (intercepto/inclinação regularizados), janelas3/6/12, árvores menores, treinamento conjunto com o mesmo material e gerador A preservado. Não se promove associação residual como ganho confirmado.')
    rows=candidates();atomic_json(ROOT/'configs/v3/cycle2_candidates.json',rows);CorrectionCampaign().development(rows,'cycle2')
    joint=[dict(name='joint_'+name,groups=groups,penalty=40,months=3,kind='joint',source_A='A_tree') for name,groups in [('v2',['v2']),('aggregate_trajectory',['aggregate_trajectory','missing']),('sensor_now',['sensor_now','missing']),('sensor_trajectory',['sensor_trajectory','missing']),('coherence',['aggregate_trajectory','coherence','missing'])]]
    atomic_json(ROOT/'configs/v3/joint_candidates.json',joint);JointCampaign().development(joint,'joint')
    compatible=[dict(name='compatible_'+name,groups=groups,kind='linear',calendar=False,penalty=1000,leaves=3,iterations=100,months=3,source_A='A_tree') for name,groups in [('score',[]),('weather',['v2']),('trajectory',['sensor_trajectory','coherence','missing'])]]
    atomic_json(ROOT/'configs/v3/compatible_candidates.json',compatible);CompatibleCampaign().development(compatible,'compatible')

if __name__=='__main__':main()
