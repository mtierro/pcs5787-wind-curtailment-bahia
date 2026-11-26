"""Campanha sequencial com cache identificado por dados, código e split."""
import argparse
import json
import time
import resource
import warnings
import fcntl
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,precision_score,recall_score,brier_score_loss,log_loss,confusion_matrix,mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
from ..common import ROOT,sha256
from .common import P,DATA,OUT,CACHE,REASONS,folds,cutoff,digest,atomic_json,atomic_frame,population,training_mask,note
from .learners import make_model


def metrics(y,p,threshold=.5):
    y=np.asarray(y).astype(int);p=np.asarray(p);yh=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,yh,labels=[0,1]).ravel()
    return dict(n=len(y),positive=int(y.sum()),prevalence=float(y.mean()),mean_p=float(p.mean()),ap=average_precision_score(y,p) if y.sum() else np.nan,roc_auc=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,f1=f1_score(y,yh,zero_division=0),precision=precision_score(y,yh,zero_division=0),recall=recall_score(y,yh,zero_division=0),brier=brier_score_loss(y,p),log_loss=log_loss(y,p,labels=[0,1]),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),threshold=threshold)


def numeric_columns(spec):
    groups=json.loads((ROOT/'configs/v2/feature_groups.json').read_text())
    cols=[]
    for name in spec.get('features',[]):cols+=groups[name]
    if spec.get('variant','A') in ['A','C']:cols=[]
    if spec.get('variant') in ['C','D']:cols+=groups['context'+spec.get('context_suffix','')]
    if spec.get('solar'):cols+=groups['solar']
    cols+=spec.get('extra_columns',[])
    return [c for c in dict.fromkeys(cols) if c not in spec.get('exclude_columns',[])]


class Campaign:
    def __init__(self):
        self.base=pd.read_parquet(DATA/'analytical.parquet').sort_values(['time_local','region']).reset_index(drop=True)
        self.dataset=sha256(DATA/'analytical.parquet')
        self.code={str(p.name):sha256(p) for p in [ROOT/'src/pcs5787/v2/learners.py',ROOT/'src/pcs5787/v2/experiments.py',ROOT/'src/pcs5787/v2/common.py']}
        self.ledger_path=OUT/'tables/experiment_ledger.csv'
        self.ledger=pd.read_csv(self.ledger_path).to_dict('records') if self.ledger_path.exists() else []
        CACHE.mkdir(parents=True,exist_ok=True)

    def run(self,spec,fold,save_model=False):
        cols=numeric_columns(spec)
        fingerprint=digest({'spec':spec,'fold':fold,'data':self.dataset,'code':self.code,'protocol':P,'columns':cols})
        ident=fingerprint[:20];meta_path=CACHE/f'{ident}.json';pred_path=CACHE/f'{ident}.parquet';model_path=ROOT/f'models/v2/{ident}.joblib'
        if meta_path.exists() and pred_path.exists() and (not save_model or model_path.exists()):
            meta=json.loads(meta_path.read_text())
            assert meta['fingerprint']==fingerprint and meta['pred_sha256']==sha256(pred_path)
            if save_model:assert meta['model_sha256']==sha256(model_path)
            return meta,pd.read_parquet(pred_path)
        start=time.monotonic()
        eligible,target=population(self.base,spec['reason'],spec.get('population','anchor_complete'))
        frame=self.base
        if spec.get('weather_shift_hours'):
            frame=self.base.copy()
            met=[c for c in cols if not c.startswith(('op','solar_','missing_'))]
            frame[met]=frame.groupby('region')[met].shift(spec['weather_shift_hours'])
        if spec.get('min_reference_mwh'):eligible &= frame.reference_mwh.ge(spec['min_reference_mwh'])
        train=frame[eligible&training_mask(frame,fold['origin'],spec['strategy'],spec.get('latency'))]
        valid=frame[eligible&frame.time_local.ge(fold['start'])&frame.time_local.lt(fold['stop'])]
        assert len(train)>0 and len(valid)>0
        assert train.time_local.max()+pd.Timedelta(hours=1+spec.get('latency',P['label_publication_latency_hours']))<=fold['start']-pd.Timedelta(hours=1)
        regression=spec['reason']=='magnitude'
        yt=train[target].to_numpy(dtype=float if regression else int);yv=valid[target].to_numpy(dtype=float if regression else int)
        weight=None
        if spec['strategy']=='half_life180':weight=np.exp2(-(cutoff(fold['origin'],spec.get('latency'))-train.time_local).dt.total_seconds().to_numpy()/(86400*180))
        model=make_model(spec,cols,regression)
        fitkwargs={'model__sample_weight':weight} if weight is not None else {}
        if spec['algorithm']=='lightgbm':fitkwargs['model__categorical_feature']=list(range(6 if spec.get('calendar','strong')=='strong' else 4))
        failure=''
        with threadpool_limits(limits=4),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if regression or len(np.unique(yt))==2:
                model.fit(train,yt,**fitkwargs)
                p=model.predict(valid) if regression else model.predict_proba(valid)[:,1]
                stride=max(1,len(train)//12000);training_p=model.predict(train.iloc[::stride]) if regression else model.predict_proba(train.iloc[::stride])[:,1]
                training_score=mean_absolute_error(yt[::stride],training_p) if regression else average_precision_score(yt[::stride],training_p)
            else:
                p=np.full(len(valid),(yt.sum()+.5)/(len(yt)+1));training_score=np.nan;failure='one_class_training_constant_prior'
        if regression:
            p=np.clip(p,0,1);m=dict(n=len(yv),mae=mean_absolute_error(yv,p),rmse=np.sqrt(mean_squared_error(yv,p)),mean_y=yv.mean(),mean_p=p.mean())
        else:m=metrics(yv,p)
        pred=valid[['time_local','region']].copy();pred['y']=yv;pred['p']=p;pred['threshold']=.5;pred['run_id']=ident
        atomic_frame(pred,pred_path)
        meta={'run_id':ident,'fingerprint':fingerprint,'spec':spec,'fold':{k:str(v) for k,v in fold.items()},'columns':cols,'n_train':len(train),'positive_train':int(np.sum(yt)) if not regression else None,'train_start':str(train.time_local.min()),'train_end':str(train.time_local.max()),'training_sample_score':training_score,'metrics':m,'seconds':time.monotonic()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'status':'done','note':failure,'warnings':sorted(set(str(w.message) for w in caught)),'pred_sha256':sha256(pred_path)}
        if save_model and not failure:
            tmp=model_path.with_suffix('.tmp');joblib.dump(model,tmp,compress=3);tmp.replace(model_path);meta['model_sha256']=sha256(model_path)
        atomic_json(meta_path,meta)
        row={'run_id':ident,'family':spec['family'],'question':spec.get('question','H1/H2'),'reason':spec['reason'],'population':spec.get('population','anchor_complete'),'dataset_sha256':self.dataset,'variant':spec.get('variant','A'),'features':json.dumps(spec.get('features',[])),'fold':fold['name'],'strategy':spec['strategy'],'algorithm':spec['algorithm'],'params':json.dumps(spec.get('params',{}),sort_keys=True),'seed':spec.get('seed',5787),'n_train':len(train),'train_start':meta['train_start'],'train_end':meta['train_end'],'training_sample_score':training_score,**m,'seconds':meta['seconds'],'peak_rss_mib':meta['peak_rss_mib'],'status':'done','exclusion_reason':failure,'warnings':' | '.join(meta['warnings'])}
        self.ledger=[r for r in self.ledger if r['run_id']!=ident]+[row]
        atomic_frame(pd.DataFrame(self.ledger),self.ledger_path)
        score=m.get('ap',m.get('mae'));print(f'{len(self.ledger)} {spec["family"]} {spec["reason"]} {spec.get("variant")} {spec["strategy"]} {spec["algorithm"]} {fold["name"]}: {score:.5f}; {meta["seconds"]:.1f}s',flush=True)
        return meta,pred


def temporal_specs():
    return [dict(family='temporal',question='H3/H1',reason=r,variant=v,features=['basic'],calendar='basic',population='anchor_complete',algorithm='logistic',params={'C':c},strategy=s,seed=5787) for r in REASONS for s in P['training_strategies'] for c in [.01,.1,1,10] for v in ['A','B']]


def select(specs,campaign,save_name):
    rows=[]
    for spec in specs:
        results=[campaign.run(spec,f)[0] for f in folds()]
        aps=[x['metrics']['ap'] for x in results]
        rows.append({'spec':spec,'mean_ap':float(np.nanmean(aps)),'std_ap':float(np.nanstd(aps)),'fold_ap':aps,'run_ids':[x['run_id'] for x in results]})
    atomic_json(ROOT/f'configs/v2/{save_name}.json',rows)
    return rows


def chosen_temporal():
    rows=json.loads((ROOT/'configs/v2/temporal_results.json').read_text());chosen={}
    for reason in REASONS:
        scores={}
        for strategy in P['training_strategies']:
            scores[strategy]=np.mean([max(r['mean_ap'] for r in rows if r['spec']['reason']==reason and r['spec']['strategy']==strategy and r['spec']['variant']==v) for v in ['A','B']])
        strategy=max(scores,key=scores.get)
        chosen[reason]={'strategy':strategy,'scores':scores,'variants':{v:max([r for r in rows if r['spec']['reason']==reason and r['spec']['strategy']==strategy and r['spec']['variant']==v],key=lambda x:x['mean_ap']) for v in ['A','B']}}
    atomic_json(ROOT/'configs/v2/temporal_selected.json',chosen)
    return chosen


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['temporal'],default='temporal');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    with (ROOT/'.scratch/v2/campaign.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=Campaign();specs=temporal_specs()
        atomic_json(ROOT/'configs/v2/temporal_grid.json',specs)
        if args.smoke:c.run(specs[0],folds()[0]);return
        select(specs,c,'temporal_results');selected=chosen_temporal();note('R2 temporal',f'Triagem completa de quatro estratégias × quatro C × A/B × três motivos × três folds. Seleção compartilhada por par: { {r:s["strategy"] for r,s in selected.items()} }. Próximo: representação/modelos/contexto e refinamento. Externo ainda não avaliado.')


if __name__=='__main__':main()
