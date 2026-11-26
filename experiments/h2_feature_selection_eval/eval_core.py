"""Historical representation, clock and metrics; frozen columns only."""
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from audit_eval import HERE,ROOT,DEV,digest,sha

sys.path.insert(0,str(ROOT/'src'))
from pcs5787.v2.learners import make_model
from pcs5787.v2.common import population,training_mask,cutoff as original_cutoff
from pcs5787.v2.calibration import available_history,decision_threshold
from pcs5787.v2.evaluation import WeightedAP

CONFIG=json.loads((HERE/'protocol.json').read_text())
FROZEN=json.loads((ROOT/CONFIG['frozen_selection']).read_text())
GROUPS=json.loads((ROOT/'configs/v2/feature_groups.json').read_text())
FEATURES=sum([GROUPS[g] for g in ['basic','engineered','other_weather','spatial']],[])
CALENDAR=['region','hour','weekday','month','region_hour','region_month']
VARIANTS=CONFIG['variants'];REDUCED=[r['variant'] for r in FROZEN['configurations']]
SPECS={r['variant']:r for r in FROZEN['configurations']}
KEYS=['time_local','region']
FINAL_IDS=[('54ca092bcc01b02c2097','2ebe49d5e5d6e7cdbc2b'),('d9ecad6f5877a9366330','a64f59db40ea883738f8'),
           ('9b40f8cb39d3ea3770d2','6b56746b5ad9e01c33a6'),('22c5e9a7b13440b2c722','0fe77d369254ea9cc4fa'),
           ('2ca2134204263d25fc5b','28b4f9c60e5fc644f75a'),('3c5196e98b4d2f39495b','dcc0d53e6ac8e1fbcdda')]


def stamp(value):
    t=pd.Timestamp(value)
    return t.tz_localize(CONFIG['timezone']) if t.tz is None else t.tz_convert(CONFIG['timezone'])


START=stamp(CONFIG['start']);STOP=stamp(CONFIG['stop_exclusive'])
ORIGINS=[stamp(o) for o in CONFIG['origins']]


def cutoff(origin):return original_cutoff(stamp(origin).strftime('%Y-%m-%d'))


def columns(variant):
    if variant=='A':return []
    if variant=='B':return FEATURES.copy()
    return SPECS[variant]['features'].copy()


def spec(variant):
    assert variant in REDUCED,'Only reduced models may be fitted in this mission'
    return SPECS[variant]['parameters']


def fingerprint(frame,target='y_ENE_02',probability=False):
    cols=KEYS+[target]+(['p'] if probability else [])
    return digest(pd.util.hash_pandas_object(frame[cols],index=False).tolist())


def load_base():
    lower=cutoff(START)-pd.DateOffset(months=3)
    path=ROOT/'data/processed/v2/analytical.parquet'
    assert sha(path)==FROZEN['semantic_data_access']['source_sha256']
    base=pd.read_parquet(path,columns=KEYS+['y_ENE_02']+[c for c in CALENDAR if c!='region']+FEATURES,
                         filters=[('time_local','>=',lower),('time_local','<',STOP)])
    base=base.sort_values(KEYS).reset_index(drop=True)
    assert not base.duplicated(KEYS).any() and not np.isinf(base[FEATURES].to_numpy()).any()
    assert base.groupby('time_local').size().eq(8).all()
    assert len(base)==int((STOP-lower)/pd.Timedelta(hours=1))*8
    base['row_id']=np.arange(len(base))
    return base


def split(base,origin):
    origin=stamp(origin);assert origin in ORIGINS
    eligible,target=population(base,'ENE','spatial_recent')
    assert target=='y_ENE_02'
    tr=base.loc[eligible & training_mask(base,origin.strftime('%Y-%m-%d'),'window3')]
    va=base.loc[eligible & base.time_local.ge(origin)&base.time_local.lt(origin+pd.DateOffset(months=1))]
    assert len(tr) and len(va)
    assert tr.time_local.min()>=cutoff(origin)-pd.DateOffset(months=3)
    assert tr.time_local.max()==origin-pd.Timedelta(hours=170)
    assert tr.time_local.max()+pd.Timedelta(hours=169)==origin-pd.Timedelta(hours=1)
    assert set(tr.row_id).isdisjoint(va.row_id)
    assert not tr[KEYS].reset_index(drop=True).equals(va[KEYS].reset_index(drop=True))
    return tr,va


def load_reference():
    ref=pd.read_parquet(ROOT/'data/processed/v2/final_predictions.parquet',filters=[
        ('protocol','==','P2'),('reason','==','ENE'),('population','==','spatial_recent'),
        ('algorithm','==','lightgbm'),('seed','==',5787),('variant','in',['A','B'])])
    assert ref.calibration.eq('none').all() and np.array_equal(ref.p,ref.p_raw)
    assert set(ref.run_id)=={r for pair in FINAL_IDS for r in pair}
    ref['origin']=ref.time_local.dt.strftime('%Y-%m')
    for v in ['A','B']:
        x=ref.loc[ref.variant.eq(v)]
        assert len(x)==34811 and x.y.sum()==11521 and x.region.nunique()==8
        assert not x.duplicated(KEYS).any() and x.time_local.min()>=START and x.time_local.max()<STOP
    return ref


def warm_start(variant):
    if variant in ['A','B']:
        df=pd.read_parquet(ROOT/'data/processed/v2/selected_oof.parquet',filters=[
            ('panel','==','quad'),('reason','==','ENE'),('population','==','spatial_recent'),
            ('variant','==',variant),('fold','==','F3'),('seed','==',5787)])
        source='historical quad F3 OOF'
    else:
        df=pd.read_parquet(DEV/'predictions.parquet',filters=[('variant','==',variant)])
        source='2B-1 monthly OOF; adaptive subsets for top5, fixed spatial for C_spatial'
    assert df.time_local.max()<cutoff(START) and not df.duplicated(KEYS).any()
    return df[KEYS+['y','p']].sort_values(KEYS).reset_index(drop=True),source


def threshold(pool,origin):
    h=available_history(pool,stamp(origin).strftime('%Y-%m-%d')).sort_values(KEYS).reset_index(drop=True)
    assert len(h)==0 or h.time_local.max()+pd.Timedelta(hours=169)<=stamp(origin)-pd.Timedelta(hours=1)
    value=decision_threshold(h.y.to_numpy(),h.p.to_numpy())
    return value,dict(n=len(h),first=str(h.time_local.min()) if len(h) else None,
                      last=str(h.time_local.max()) if len(h) else None,
                      last_label_available=str(h.time_local.max()+pd.Timedelta(hours=169)) if len(h) else None,
                      history_sha256=fingerprint(h,'y',True),fallback=bool(len(h)<100 or h.y.sum()==0))


def matrix(model,frame,variant):
    cols=columns(variant);prep=model.named_steps['representation'];clf=model.named_steps['model']
    assert prep.cat_==CALENDAR and list(prep.numeric)==cols and prep.tree and not prep.anomaly
    assert list(clf.classes_)==[0,1] and clf.booster_.num_trees()==180
    assert clf.booster_.params['categorical_column']==list(range(6))
    params=SPECS['C_top5']['effective_estimator_parameters'].copy();params['reg_lambda']=40 if variant=='A' else 10
    assert clf.get_params()==params
    z=prep.transform(frame);k=len(cols)
    assert z.shape==(len(frame),6+2*k)
    assert np.array_equal(z[:,6:6+k],frame[cols].to_numpy(float),equal_nan=True)
    assert np.array_equal(z[:,6+k:],~np.isfinite(z[:,6:6+k]))
    return z


def score(y,p,t):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=p>=np.asarray(t)
    tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tn=int(((y==0)&~pred).sum())
    return dict(n=len(y),positive=int(y.sum()),prevalence=float(y.mean()),ap=float(average_precision_score(y,p)),
                brier=float(np.mean((p-y)**2)),roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
                precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,specificity=tn/(tn+fp) if tn+fp else 0.,
                alert_fraction=float(pred.mean()),alert_region_hours=int(pred.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
                mean_p=float(p.mean()))


def align(pred,variants=VARIANTS):
    ref=pred.loc[pred.variant.eq('B')].sort_values(KEYS).reset_index(drop=True)
    result={}
    assert len(ref)==34811 and ref.y.sum()==11521
    for v in variants:
        f=pred.loc[pred.variant.eq(v)].sort_values(KEYS).reset_index(drop=True)
        assert len(f)==len(ref) and not f.duplicated(KEYS).any()
        for c in KEYS+['y','origin']:assert np.array_equal(f[c],ref[c]),(v,c)
        result[v]=f
    return result
