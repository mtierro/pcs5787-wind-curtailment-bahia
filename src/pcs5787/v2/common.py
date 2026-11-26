"""I/O atômico, identidade de execução e regras temporais compartilhadas."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import pyarrow as pa
from ..common import ROOT, sha256

P=json.loads((ROOT/'configs/v2/protocol.json').read_text())
pa.set_cpu_count(P['threads'])
OUT=ROOT/'reports/v2'
DATA=ROOT/'data/processed/v2'
CACHE=ROOT/'.scratch/v2/cache'
REASONS=['ENE','CNF','REL']
V1MET=[f'wind_lag{k}' for k in [1,3,6,24]]+[f'gust_lag{k}' for k in [1,24]]+['wind_mean24','gust_max24']


def stamp(x):return pd.Timestamp(x,tz=P['timezone'])


def cutoff(origin,latency=None):
    return stamp(origin) - pd.Timedelta(hours=(P['label_publication_latency_hours'] if latency is None else latency)+P['forecast_horizon_hours'])


def folds():
    result=[]
    for f in P['development_folds']:
        end=stamp(f['end'])
        if f.get('end_limit'):end=min(end,cutoff(P['final_origin']))
        result.append({**f,'start':stamp(f['origin']),'stop':end,'train_stop':cutoff(f['origin'])})
    return result


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n');os.replace(tmp,path)


def atomic_frame(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.stem+'.tmp'+path.suffix)
    if path.suffix=='.parquet':frame.to_parquet(tmp,index=False)
    else:frame.to_csv(tmp,index=False)
    os.replace(tmp,path)


def note(stage,message):
    path=OUT/'STATUS.md'
    with path.open('a') as f:f.write(f'\n## {datetime.now(timezone.utc).isoformat(timespec="seconds")} — {stage}\n\n{message}\n')


def training_mask(frame,origin,strategy,latency=None):
    end=cutoff(origin,latency)
    mask=frame.time_local.lt(end)
    if strategy.startswith('window'):
        mask &= frame.time_local.ge(end-pd.DateOffset(months=int(strategy[6:])))
    return mask


def population(frame,reason,name='anchor_complete'):
    target={'ENE':'y_ENE_02','CNF':'y_CNF_02','REL':'y_REL_02','onset':'onset_ENE','end':'end_ENE','widespread':'widespread_ENE','magnitude':'fraction_total'}[reason]
    mask=frame[target].notna()
    if name=='v1_common':mask &= frame[[f'y_{r}_02' for r in REASONS]+V1MET].notna().all(axis=1)
    elif name=='anchor_complete':mask &= frame[V1MET].notna().all(axis=1)
    elif name=='anchor_recent':mask &= frame.wind_lag1.notna()
    elif name=='spatial_recent':mask &= frame.sp_wind.notna()
    elif name!='ons':raise ValueError(name)
    return mask,target
