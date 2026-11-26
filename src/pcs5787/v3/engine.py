"""Ajustes cronológicos com cache e metadados; nunca escreve na V2."""
import json,time,resource,warnings
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from ..v2.learners import make_model
from ..v2.experiments import metrics,numeric_columns
from .common import ROOT,DATA,OUT,CACHE,MODELS,P,atomic_json,atomic_frame,digest,sha256,stamp,cutoff,population,training_mask,note

class Engine:
    def __init__(self):
        self.base=pd.read_parquet(DATA/'analytical.parquet')
        self.dataset=sha256(DATA/'analytical.parquet')
        self.groups=json.loads((ROOT/'configs/v3/feature_groups.json').read_text())
        self.code={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'src/pcs5787/v3/engine.py',ROOT/'src/pcs5787/v2/learners.py',ROOT/'src/pcs5787/v2/common.py']}
        CACHE.mkdir(parents=True,exist_ok=True);MODELS.mkdir(exist_ok=True)
        self.ledger_path=OUT/'tables/base_ledger.csv'
        self.ledger=pd.read_csv(self.ledger_path).to_dict('records') if self.ledger_path.exists() else []

    def spec(self,name,reason='ENE'):
        frozen=json.loads((ROOT/'configs/v2/finalists_frozen.json').read_text())
        panel='best_AB' if name=='A_log' else 'quad';v='B' if name=='B_v2' else 'A'
        r=next(x['spec'] for x in frozen[panel] if x['spec']['reason']==reason and x['spec']['population']=='spatial_recent' and x['spec']['variant']==v).copy()
        r['seed']=P['seed'];return r

    def base_model(self,name,origin,reason='ENE',seed=5787):
        spec=self.spec(name,reason);spec['seed']=seed;cols=numeric_columns(spec)
        identity=dict(name=name,spec=spec,origin=origin,data=self.dataset,code=self.code,columns=cols)
        ident=digest(identity)[:24];path=MODELS/f'base_{ident}.joblib';meta_path=MODELS/f'base_{ident}.json'
        if path.exists() and meta_path.exists():
            meta=json.loads(meta_path.read_text());assert sha256(path)==meta['model_sha256'];return joblib.load(path),meta
        start=time.monotonic();mask,target=population(self.base,reason,'spatial_recent');train=self.base[mask&training_mask(self.base,origin,spec['strategy'])]
        y=train[target].to_numpy(int);model=make_model(spec,cols)
        kwargs={}
        if spec['strategy']=='half_life180':kwargs['model__sample_weight']=np.exp2(-(cutoff(origin)-train.time_local).dt.total_seconds().to_numpy()/(86400*180))
        if spec['algorithm']=='lightgbm':kwargs['model__categorical_feature']=list(range(6 if spec['calendar']=='strong' else 4))
        assert train.time_local.max()+pd.Timedelta(hours=169)<=stamp(origin)-pd.Timedelta(hours=1)
        with threadpool_limits(limits=4),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(train,y,**kwargs)
        joblib.dump(model,path,compress=3)
        meta=dict(identity=identity,id=ident,train_first=str(train.time_local.min()),train_last=str(train.time_local.max()),n_train=len(train),positive=int(y.sum()),seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,model_sha256=sha256(path),warnings=sorted(set(str(w.message) for w in caught)))
        atomic_json(meta_path,meta)
        row={k:v for k,v in meta.items() if k!='identity'};row.update(name=name,origin=origin,reason=reason,seed=seed)
        self.ledger=[r for r in self.ledger if r['id']!=ident]+[row];atomic_frame(pd.DataFrame(self.ledger),self.ledger_path)
        print('Base',name,reason,origin,len(train),f'{meta["seconds"]:.2f}s',flush=True)
        return model,meta

    def predict_base(self,name,origin,end,reason='ENE',seed=5787,allow_final=False):
        if not allow_final:assert stamp(end)<=stamp('2026-04-01')
        model,meta=self.base_model(name,origin,reason,seed)
        ident=digest(dict(model=meta['id'],end=end))[:24];path=CACHE/f'base_pred_{ident}.parquet'
        if path.exists():return pd.read_parquet(path)
        mask,target=population(self.base,reason,'spatial_recent');v=self.base[mask&self.base.time_local.ge(stamp(origin))&self.base.time_local.lt(stamp(end))]
        with threadpool_limits(limits=4):prob=model.predict_proba(v)[:,1]
        d=v[['region','time_local']].copy();d['y']=v[target].to_numpy();d['p']=prob;d['origin']=origin;d['name']=name;d['reason']=reason;d['model_id']=meta['id'];d['train_last']=meta['train_last']
        atomic_frame(d,path);return d

    def oof(self):
        pieces=[]
        for origin in pd.date_range(P['monthly_oof_first'],'2026-03-01',freq='MS'):
            end=min(origin+pd.DateOffset(months=1),pd.Timestamp(P['development_end_exclusive']))
            for name in ['A_tree','A_log','B_v2']:pieces.append(self.predict_base(name,str(origin.date()),str(end)))
        out=pd.concat(pieces,ignore_index=True)
        assert not out.duplicated(['region','time_local','name','reason']).any()
        atomic_frame(out,DATA/'base_oof.parquet');atomic_json(OUT/'tables/base_oof_identity.json',dict(rows=len(out),sha256=sha256(DATA/'base_oof.parquet'),dataset_sha256=self.dataset,latest=str(out.time_local.max()),future_labels_excluded=True))
        note('C3 OOF',f'{len(out)} previsões mensais temporais A árvore, A logística e B V2; alvos nunca usados no respectivo ajuste. Limite {out.time_local.max()}. Diagnóstico/offset a seguir.')

if __name__=='__main__':Engine().oof()
