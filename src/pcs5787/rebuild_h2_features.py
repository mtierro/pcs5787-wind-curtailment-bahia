"""Pure reconstruction of 51 historical inputs from published station data/mapping.

Equivalent equations to features.py and v2/data.py. No labels, fitting, downloads,
or writes. Historical geometry and data masks are consumed, never reselected.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from .features import historical_features

def rebuild(data_root):
    root=Path(data_root)/'observational'
    w=pd.read_parquet(root/'station_hourly.parquet')
    mapping=pd.read_csv(root/'region_station_links.csv')
    w['time_local']=w.time_utc.dt.tz_convert('America/Bahia')
    grid=pd.date_range('2022-01-01','2026-10-01',inclusive='left',freq='h',tz='America/Bahia')
    mats={c:w.pivot(index='time_local',columns='station',values=c).reindex(grid) for c in ['vento_ms','rajada_ms','wind_u','wind_v','temperatura_c']}
    frames=[]
    for region,links in mapping.groupby('region',sort=True):
        anchor=w[w.station==region].set_index('time_local').sort_index()
        b=historical_features(anchor,delay_hours=2).reindex(grid)
        x=anchor.reindex(grid);v=x.vento_ms.shift(2);g=x.rajada_ms.shift(2)
        b['wind_ramp6']=v-v.shift(6);b['wind_ramp24']=v-v.shift(24)
        b['wind_cubic_proxy']=v**3;b['gust_excess']=g-v;b['gust_ratio']=g/(v+1)
        for n in [6,24,72]:
            roll=v.rolling(n,min_periods=max(3,n//2))
            for kind in ['mean','std','min','max']:b[f'wind_{kind}{n}_partial']=getattr(roll,kind)()
            b[f'wind_coverage{n}']=v.notna().rolling(n,min_periods=n).mean()
        for c in ['temperatura_c','umidade_pct','pressure_hpa','radiacao_kjm2','precipitacao_mm','direction_sin','direction_cos','wind_u','wind_v']:
            b[c+'_lag2']=x[c].shift(2)
        s=pd.DataFrame(index=grid);weights=links.weight.to_numpy();distance=links.distance_km.to_numpy()
        for var,label in [('vento_ms','wind'),('rajada_ms','gust'),('wind_u','u'),('wind_v','v'),('temperatura_c','temperature')]:
            z=mats[var].reindex(columns=links.station).to_numpy();valid=np.isfinite(z);ww=valid*weights;den=ww.sum(axis=1)
            mean=np.divide(np.nansum(z*weights,axis=1),den,out=np.full(len(grid),np.nan),where=den>0)
            s['sp_'+label]=mean
            if label=='wind':
                s['sp_count']=valid.sum(axis=1)
                s['sp_distance']=np.divide((ww*distance).sum(axis=1),den,out=np.full(len(grid),np.nan),where=den>0)
                s['sp_wind_std']=np.sqrt(np.divide(np.nansum(weights*(z-mean[:,None])**2,axis=1),den,out=np.full(len(grid),np.nan),where=den>0))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    s['sp_wind_q25']=np.nanquantile(z,.25,axis=1);s['sp_wind_q75']=np.nanquantile(z,.75,axis=1)
        s=s.shift(2);s['sp_ramp6']=s.sp_wind-s.sp_wind.shift(6)
        s['sp_mean24']=s.sp_wind.rolling(24,min_periods=12).mean();s['sp_std24']=s.sp_wind.rolling(24,min_periods=12).std()
        s['sp_gust_excess']=s.sp_gust-s.sp_wind
        b=b.join(s);b['region']=region;frames.append(b.rename_axis('time_local').reset_index())
    return pd.concat(frames,ignore_index=True)
