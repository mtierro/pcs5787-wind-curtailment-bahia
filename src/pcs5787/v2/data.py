"""Atributos V2 sobre alvos preservados: sensores, contexto e tarefas adicionais."""
import json
import re
import warnings
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import requests
from ..common import ROOT,SOURCE,sha256
from ..audit import parse_numeric
from ..geography import distance_km
from ..integration import energy_bounds
from .common import P,DATA,OUT,V1MET,atomic_json,atomic_frame,stamp,note


def download_context():
    folder=ROOT/'data/raw/v2/ons_balance';folder.mkdir(parents=True,exist_ok=True)
    endpoint='https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/balanco_energia_subsistema_ho/'
    manifest_path=ROOT/'configs/v2/context_sources.json'
    old=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    byname={r['file']:r for r in old};rows=[];frames=[]
    for name in [f'BALANCO_ENERGIA_SUBSISTEMA_{y}.parquet' for y in range(2022,2027)]+['DicionarioDados_Balanco_Energia_Subsistema.json']:
        p=folder/name
        if not p.exists():
            r=requests.get(endpoint+name,timeout=90);r.raise_for_status();tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(r.content);tmp.replace(p)
        if name in byname:assert sha256(p)==byname[name]['sha256']
        rows.append({'file':name,'path':str(p.relative_to(ROOT)),'url':endpoint+name,'sha256':sha256(p),'bytes':p.stat().st_size,'accessed_utc':byname.get(name,{}).get('accessed_utc',datetime.now(timezone.utc).isoformat()),'license':'CC-BY no catálogo; dados sujeitos a revisões','availability':'cenário retrospectivo de latência; sem carimbo histórico de publicação'})
        if name.endswith('.parquet'):frames.append(pd.read_parquet(p))
    atomic_json(manifest_path,rows)
    x=pd.concat(frames,ignore_index=True)
    atomic_json(OUT/'tables/context_schema.json',{'columns':{c:str(t) for c,t in x.dtypes.items()},'rows_downloaded':len(x),'subsystems':x.id_subsistema.unique().tolist()})
    x=x[x.id_subsistema.eq('NE')].copy()
    for col in [c for c in x if c.startswith('val_')]:
        x[col]=pd.to_numeric(x[col].astype('string').str.replace(',','.',regex=False),errors='raise')
    x['time_local']=pd.to_datetime(x.din_instante).dt.tz_localize('America/Bahia')
    x=x[x.time_local.ge(stamp('2022-01-01'))&x.time_local.lt(stamp('2026-10-01'))]
    assert not x.time_local.duplicated().any()
    return x.set_index('time_local').sort_index()


def weather_extended():
    path=DATA/'weather_extended.parquet'
    w=pd.read_parquet(ROOT/'data/interim/weather.parquet')
    pieces=[]
    for p in sorted((SOURCE/'data/interim/inmet_bahia').rglob('*.parquet')):
        station=re.search(r'_(A\d{3})_',p.name).group(1)
        cols=['Data','Hora UTC','VENTO, DIREÇÃO HORARIA (gr) (° (gr))','PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)']
        z=pd.read_parquet(p,columns=cols)
        t=pd.to_datetime(z.Data.str.replace('/','-')+' '+z['Hora UTC'].str[:2]+':00:00',utc=True)
        direction=parse_numeric(z[cols[2]]);pressure=parse_numeric(z[cols[3]])
        pieces.append(pd.DataFrame({'station':station,'time_utc':t,'direction_deg':direction.where(direction.between(0,360)),'pressure_hpa':pressure.where(pressure.between(500,1100))}))
    ext=pd.concat(pieces,ignore_index=True)
    w=w.merge(ext,on=['station','time_utc'],validate='one_to_one')
    theta=np.radians(w.direction_deg)
    w['wind_u']=-w.vento_ms*np.sin(theta);w['wind_v']=-w.vento_ms*np.cos(theta)
    w['direction_sin']=np.sin(theta);w['direction_cos']=np.cos(theta)
    w['radiacao_kjm2']=w.radiacao_kjm2.where(w.radiacao_kjm2.ge(0))
    coords=pd.read_csv(ROOT/'reports/tables/inmet_station_year.csv')
    anchors=coords[coords.year.le(2024)].groupby('station')[['latitude','longitude']].median()
    for row in coords.itertuples():
        if row.station in anchors.index:
            a=anchors.loc[row.station]
            if distance_km(row.latitude,row.longitude,a.latitude,a.longitude)>5:
                mask=w.station.eq(row.station)&w.time_utc.dt.year.eq(row.year)
                w.loc[mask,w.select_dtypes('number').columns]=np.nan
    atomic_frame(w,path)
    return w


def lagged_weather(w,grid):
    w=w.reindex(grid)
    out=pd.DataFrame(index=grid)
    v=w.vento_ms.shift(2);g=w.rajada_ms.shift(2)
    out['wind_ramp6']=v-v.shift(6);out['wind_ramp24']=v-v.shift(24)
    out['wind_cubic_proxy']=v**3
    out['gust_excess']=g-v
    out['gust_ratio']=g/(v+1) # regularizado; não razão singular próximo de calmaria
    for window in [6,24,72]:
        roll=v.rolling(window,min_periods=max(3,window//2))
        for name,series in [('mean',roll.mean()),('std',roll.std()),('min',roll.min()),('max',roll.max())]:out[f'wind_{name}{window}_partial']=series
        out[f'wind_coverage{window}']=v.notna().rolling(window,min_periods=window).mean()
    for col in ['temperatura_c','umidade_pct','pressure_hpa','radiacao_kjm2','precipitacao_mm','direction_sin','direction_cos','wind_u','wind_v']:
        out[col+'_lag2']=w[col].shift(2)
    return out


def solar_geometry(index,latitude,longitude):
    """Aproximação analítica determinística, sem dados meteorológicos ou y."""
    day=index.dayofyear.to_numpy();hour=index.hour.to_numpy()
    angle=2*np.pi*(day-1)/365.25
    decl=.006918-.399912*np.cos(angle)+.070257*np.sin(angle)-.006758*np.cos(2*angle)+.000907*np.sin(2*angle)-.002697*np.cos(3*angle)+.00148*np.sin(3*angle)
    equation=229.18*(.000075+.001868*np.cos(angle)-.032077*np.sin(angle)-.014615*np.cos(2*angle)-.040849*np.sin(2*angle))
    solar_hour=hour+(equation+4*longitude+180)/60
    ha=np.radians(15*(solar_hour-12));lat=np.radians(latitude)
    elevation=np.sin(lat)*np.sin(decl)+np.cos(lat)*np.cos(decl)*np.cos(ha)
    return pd.DataFrame({'solar_coszenith':np.clip(elevation,0,1),'solar_hour_sin':np.sin(ha),'solar_hour_cos':np.cos(ha)},index=index)


def spatial_weather(weather,grid,geo):
    coords=pd.read_csv(ROOT/'reports/tables/inmet_station_year.csv')
    anchor=coords[coords.year.le(2024)].groupby('station')[['latitude','longitude']].median()
    w=weather.copy()
    for row in coords.itertuples():
        if row.station in anchor.index:
            a=anchor.loc[row.station]
            if distance_km(row.latitude,row.longitude,a.latitude,a.longitude)>5:
                mask=w.station.eq(row.station)&w.time_utc.dt.year.eq(row.year)
                w.loc[mask,w.select_dtypes('number').columns]=np.nan
    w['time_local']=w.time_utc.dt.tz_convert('America/Bahia')
    matrices={var:w.pivot(index='time_local',columns='station',values=var).reindex(grid) for var in ['vento_ms','rajada_ms','wind_u','wind_v','temperatura_c']}
    frames={};mapping=[]
    for region,g in geo[geo.period.le('2024_12')].groupby('region'):
        # Cada conjunto conta uma vez na geometria de desenvolvimento.
        centroid=g.drop_duplicates('id_ons')[['latitude','longitude']].mean()
        d=distance_km(centroid.latitude,centroid.longitude,anchor.latitude,anchor.longitude).sort_values()
        d=d[d.le(P['spatial_radius_km'])].iloc[:P['spatial_candidates_max_stations']]
        assert len(d)>0
        weights=1/np.maximum(d.to_numpy(),10)**2
        for station,dist in d.items():mapping.append(dict(region=region,station=station,distance_km=dist,weight=1/max(dist,10)**2,centroid_lat=centroid.latitude,centroid_lon=centroid.longitude))
        out=pd.DataFrame(index=grid)
        for var,name in [('vento_ms','wind'),('rajada_ms','gust'),('wind_u','u'),('wind_v','v'),('temperatura_c','temperature')]:
            z=matrices[var].reindex(columns=d.index).to_numpy();valid=np.isfinite(z)
            ww=valid*weights;den=ww.sum(axis=1)
            mean=np.divide(np.nansum(z*weights,axis=1),den,out=np.full(len(grid),np.nan),where=den>0)
            out[f'sp_{name}']=mean
            if var=='vento_ms':
                out['sp_count']=valid.sum(axis=1)
                out['sp_distance']=np.divide((ww*d.to_numpy()).sum(axis=1),den,out=np.full(len(grid),np.nan),where=den>0)
                out['sp_wind_std']=np.sqrt(np.divide(np.nansum(weights*(z-mean[:,None])**2,axis=1),den,out=np.full(len(grid),np.nan),where=den>0))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    out['sp_wind_q25']=np.nanquantile(z,.25,axis=1);out['sp_wind_q75']=np.nanquantile(z,.75,axis=1)
        out=out.shift(2)
        out['sp_ramp6']=out.sp_wind-out.sp_wind.shift(6)
        out['sp_mean24']=out.sp_wind.rolling(24,min_periods=12).mean()
        out['sp_std24']=out.sp_wind.rolling(24,min_periods=12).std()
        out['sp_gust_excess']=out.sp_gust-out.sp_wind
        frames[region]=out
    atomic_frame(pd.DataFrame(mapping),OUT/'tables/spatial_sensor_mapping.csv')
    return frames


def episode_labels(y):
    previous=y.shift(1);known=y.notna()&previous.notna()
    onset=((y==1)&(previous==0)).astype(float).where(known)
    end=((y==0)&(previous==1)).astype(float).where(known)
    return onset,end


def widespread_labels(base,geo):
    x=pd.read_parquet(ROOT/'data/interim/ons.parquet')
    x['period']=x.time_local.dt.strftime('%Y_%m');x=x.merge(geo,on=['id_ons','period'],validate='many_to_one')
    e=energy_bounds(x);x=pd.concat([x,e],axis=1);x['hour']=x.time_local.dt.floor('h')
    g=x.groupby(['region','hour','id_ons']).agg(ref=('reference_mwh','sum'),low=('cut_low_ENE','sum'),high=('cut_high_ENE','sum')).reset_index()
    g['pos']=g.ref.gt(0)&g.low.gt(.02*g.ref);g['negative']=g.ref.gt(0)&g.high.le(.02*g.ref)
    a=g.groupby(['region','hour']).agg(unit_positive=('pos','sum'),unit_negative=('negative','sum'),n=('id_ons','size')).reset_index().rename(columns={'hour':'time_local'})
    z=base[['region','time_local','ons_eligible']].merge(a,on=['region','time_local'],how='left',validate='one_to_one')
    lower=z.unit_positive/z.n;upper=1-z.unit_negative/z.n
    label=pd.Series(np.nan,index=z.index)
    valid=z.ons_eligible&z.n.ge(3)
    label.loc[valid&upper.lt(.5)]=0;label.loc[valid&lower.ge(.5)]=1
    return label


def main():
    source_keys={name:sha256(ROOT/name) for name in ['data/processed/v1_reference/regional_hourly.parquet','data/interim/weather.parquet','data/interim/ons.parquet','metadata/unit_region_monthly.csv','configs/v2/protocol.json']}
    context=download_context()
    weather=weather_extended()
    base=pd.read_parquet(ROOT/'data/processed/v1_reference/regional_hourly.parquet')
    geo=pd.read_csv(ROOT/'metadata/unit_region_monthly.csv')
    grid=pd.date_range('2022-01-01','2026-10-01',inclusive='left',freq='h',tz='America/Bahia')
    spatial=spatial_weather(weather,grid,geo)
    context=context.reindex(grid)
    numcols=[c for c in context if c.startswith('val_')]
    groups={'basic':V1MET,'engineered':[],'other_weather':[],'spatial':[],'context':[],'context24':[],'context720':[],'solar':['solar_coszenith','solar_hour_sin','solar_hour_cos']}
    pieces=[]
    for region,b in base.groupby('region',sort=True):
        b=b.set_index('time_local').reindex(grid)
        w=weather[weather.station.eq(region)].set_index('time_utc').sort_index();w.index=w.index.tz_convert('America/Bahia')
        f=lagged_weather(w,grid)
        b=b.join(f).join(spatial[region])
        g=geo[geo.region.eq(region)&geo.period.le('2024_12')].drop_duplicates('id_ons')
        b=b.join(solar_geometry(grid,g.latitude.mean(),g.longitude.mean()))
        b['region_hour']=b.region.astype(str)+'_'+b.hour.astype(str)
        b['region_month']=b.region.astype(str)+'_'+b.month.astype(str)
        b['fraction_total']=(b.cut_total_mwh/b.reference_mwh).where(b.ons_eligible)
        b['onset_ENE'],b['end_ENE']=episode_labels(b.y_ENE_02)
        b['previous_ENE']=b.y_ENE_02.shift(1) # somente estratificação/tarefa, jamais entrada
        for latency in [24,168,720]:
            lag=latency+2;suffix='' if latency==168 else str(latency)
            cols=[]
            for reason in ['ENE','CNF','REL']:
                name=f'op{suffix}_{reason}_past';b[name]=b[f'y_{reason}_02'].shift(lag);cols.append(name)
                name=f'op{suffix}_{reason}_mean24';b[name]=b[f'y_{reason}_02'].shift(lag).rolling(24,min_periods=18).mean();cols.append(name)
            name=f'op{suffix}_fraction';b[name]=b.fraction_total.shift(lag);cols.append(name)
            name=f'op{suffix}_reference';b[name]=np.log1p(b.reference_mwh.shift(lag));cols.append(name)
            for col in numcols:
                name=f'op{suffix}_NE_{col}';b[name]=context[col].shift(lag);cols.append(name)
            groups['context'+suffix]=cols
        groups['engineered']=[c for c in f if c.startswith(('wind_','gust_'))]
        groups['other_weather']=[c for c in f if c not in groups['engineered']]
        groups['spatial']=list(spatial[region].columns)
        pieces.append(b.rename_axis('time_local').reset_index())
    b=pd.concat(pieces,ignore_index=True).sort_values(['region','time_local']).reset_index(drop=True)
    b['widespread_ENE']=widespread_labels(b,geo)
    # Indicadores como controle separado; não preencher longas falhas com clima sintético.
    groups['missing'] = []
    for c in V1MET+groups['other_weather']+['sp_wind','sp_gust']:
        name='missing_'+c;b[name]=b[c].isna().astype(float);groups['missing'].append(name)
    assert not b.duplicated(['region','time_local']).any()
    assert np.array_equal(b[[f'y_{r}_02' for r in ['ENE','CNF','REL']]].to_numpy(),base.sort_values(['region','time_local'])[[f'y_{r}_02' for r in ['ENE','CNF','REL']]].to_numpy(),equal_nan=True)
    atomic_frame(b,DATA/'analytical.parquet')
    atomic_json(ROOT/'configs/v2/feature_groups.json',groups)
    source_keys.update({'context_sources':sha256(ROOT/'configs/v2/context_sources.json'),'data_builder':sha256(__file__)})
    atomic_json(ROOT/'configs/v2/dataset_manifest.json',{'source_hashes':source_keys,'rows':len(b),'columns':len(b.columns),'dataset_sha256':sha256(DATA/'analytical.parquet'),'generated_at_utc':datetime.now(timezone.utc).isoformat(),'external_labels_used_for_feature_learning':False})
    coverage=b.groupby(['year','region']).agg(hours=('region','size'),recent_anchor=('wind_lag1','count'),recent_multisensor=('sp_wind','count'),median_sensors=('sp_count','median')).reset_index()
    atomic_frame(coverage,OUT/'tables/v2_sensor_coverage.csv')
    note('R1','Base V2 construída preservando os rótulos V1. Meteorologia estendida, multissensor, fase solar, históricos ONS e balanço público NE com atraso. Novas tarefas de início e disseminação; grupo de colunas contemporâneas não autorizado como entrada. Manifesto de dados e grupos salvo.')
    print('Dataset V2:',b.shape,flush=True)


if __name__=='__main__':main()
