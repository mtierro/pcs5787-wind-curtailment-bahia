"""Portable validation and evaluation of the published datasets (no fitting).

These entry points consume only the two supplied datasets and the public package.
The historical representation module is preserved byte-for-byte for joblib loading.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score

KEY=['region','time_local']
CAL=['region','hour','weekday','month','region_hour','region_month']
REGIONS=['A408','A423','A426','A429','A430','A433','A439','A458']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def verify_manifest(root):
    root=Path(root);lines=(root/'MANIFEST.sha256').read_text().splitlines()
    names=[]
    for line in lines:
        expected,name=line.split('  ',1);p=root/name
        if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink():raise ValueError('Unsafe manifest path')
        if sha(p)!=expected:raise ValueError('Checksum mismatch: '+name)
        names.append(name)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256'}
    if set(names)!=actual or len(names)!=len(set(names)):raise ValueError('Manifest file set mismatch')
    return len(names)

def read_data(data_root):
    root=Path(data_root);a=root/'observational';b=root/'experiments'
    reg=pd.read_parquet(a/'regional_hourly.parquet');features=pd.read_parquet(b/'features_h2.parquet')
    if reg.duplicated(KEY).any() or features.duplicated(KEY).any():raise ValueError('Duplicate regional keys')
    if not pd.MultiIndex.from_frame(reg[KEY]).equals(pd.MultiIndex.from_frame(features[KEY])):
        raise ValueError('Dataset row order/keys differ')
    x=features.merge(reg[KEY+[c for c in reg if c not in features]],on=KEY,how='left',validate='one_to_one')
    if len(x)!=332928:raise ValueError('Unexpected population')
    return a,b,reg,features,x

def score(g):
    y=g.y.to_numpy(int);p=g.p.to_numpy(float);alert=p>=g.threshold.to_numpy(float)
    tp=int(((y==1)&alert).sum());fp=int(((y==0)&alert).sum())
    fn=int(((y==1)&~alert).sum());tn=int(((y==0)&~alert).sum())
    return dict(n=len(g),positive=int(y.sum()),prevalence=float(y.mean()),ap=float(average_precision_score(y,p)),
      brier=float(np.mean((p-y)**2)),roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
      precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
      f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,specificity=tn/(tn+fp) if tn+fp else 0.,
      alert_fraction=float(alert.mean()),tp=tp,fp=fp,fn=fn,tn=tn)

def metrics(data_root):
    b=Path(data_root)/'experiments';rows=[]
    p=pd.read_parquet(b/'evaluation/predictions.parquet')
    for v,g in p.groupby('variant',sort=False):rows.append(dict(hypothesis='H2',system=v,**score(g)))
    p=pd.read_parquet(b/'predictions/hypotheses.parquet')
    for (h,s),g in p.groupby(['hypothesis','system'],sort=False):rows.append(dict(hypothesis=h,system=s,**score(g)))
    return pd.DataFrame(rows)

def validate(data_root,models=True):
    a,b,reg,features,base=read_data(data_root);checks=[];details={}
    def check(name,cond):
        checks.append({'name':name,'passed':bool(cond)})
        if not cond:raise AssertionError(name)
    check('Dataset 1 manifest',verify_manifest(a)>0);check('Dataset 2 manifest',verify_manifest(b)>0)
    weather=pd.read_parquet(a/'station_hourly.parquet')
    check('Regional dimensions/coverage',reg.shape==(332928,31) and sorted(reg.region.unique())==REGIONS and reg.groupby('region').size().eq(41616).all())
    check('Station dimensions/uniqueness',weather.shape==(1871136,14) and weather.station.nunique()==47 and not weather.duplicated(['station','time_utc']).any())
    check('Timezones',str(reg.time_local.dt.tz)=='America/Bahia' and str(weather.time_utc.dt.tz)=='UTC')
    check('Regional time bounds',str(reg.time_local.min())=='2022-01-01 00:00:00-03:00' and str(reg.time_local.max())=='2026-09-30 23:00:00-03:00')
    dims={n:pd.read_csv(a/(n+'.csv')) for n in ['regions','stations_yearly','wind_sets_monthly','plant_membership_monthly','plants','region_station_links']}
    dkeys={'regions':['region'],'stations_yearly':['station','year'],'wind_sets_monthly':['id_ons','period'],
       'plant_membership_monthly':['id_ons_conjuntousina','ceg','period'],'plants':['ceg_norm'],'region_station_links':['region','station']}
    expected={'regions':8,'stations_yearly':225,'wind_sets_monthly':2578,'plant_membership_monthly':17712,'plants':370,'region_station_links':39}
    for n,d in dims.items():check(n+' dimensions and primary key',len(d)==expected[n] and not d.duplicated(dkeys[n]).any() and d[dkeys[n]].notna().all().all())
    check('Station foreign keys',set(weather.station)<=set(dims['stations_yearly'].station) and set(dims['region_station_links'].station)<=set(weather.station))
    check('Region foreign keys',set(dims['wind_sets_monthly'].region)<=set(REGIONS) and set(dims['region_station_links'].region)==set(REGIONS))
    m=dims['plant_membership_monthly'];u=dims['wind_sets_monthly'];plants=dims['plants']
    mkey=pd.MultiIndex.from_frame(m[['id_ons_conjuntousina','period']].rename(columns={'id_ons_conjuntousina':'id_ons'}))
    check('Membership wind-set foreign key',mkey.isin(pd.MultiIndex.from_frame(u[['id_ons','period']])).all())
    check('Membership plant foreign key',set(m.ceg)<=set(plants.ceg))
    links=dims['region_station_links'];check('Historical spatial weights',np.allclose(links.weight,1/np.maximum(links.distance_km,10)**2,rtol=1e-12,atol=1e-16))
    check('Energy eligibility',int(reg.ons_eligible.sum())==315828 and reg.loc[~reg.ons_eligible,['reference_mwh','generation_mwh','cut_total_mwh']].isna().all().all())
    check('Energy fraction',np.allclose(reg.fraction_total,reg.cut_total_mwh/reg.reference_mwh,atol=0,rtol=1e-14,equal_nan=True))
    for reason in ['ENE','CNF','REL']:
        lo=reg['cut_low_'+reason]/reg.reference_mwh;hi=reg['cut_high_'+reason]/reg.reference_mwh
        check(reason+' energy bounds',((reg['cut_low_'+reason]<=reg['cut_high_'+reason])|~reg.ons_eligible).all())
        for tau in [1,2,5]:
            target=np.full(len(reg),np.nan);valid=reg.ons_eligible & reg.reference_mwh.gt(0)
            target[valid & hi.le(tau/100)]=0;target[valid & lo.gt(tau/100)]=1
            check(f'{reason} {tau}% labels',np.array_equal(target,reg[f'y_{reason}_{tau:02d}'].to_numpy(float),equal_nan=True))
    representation=json.loads((b/'protocols/representation.json').read_text());met=representation['meteorological']
    check('51 ordered input features',len(met)==len(set(met))==51 and list(features)==KEY+[c for c in CAL if c not in KEY]+met)
    check('Preserved missing weather',features.sp_wind.isna().sum()==156 and features.wind_lag1.isna().sum()==71309 and not np.isinf(features[met].to_numpy()).any())
    check('Zero-based calendar',np.array_equal(features.month,features.time_local.dt.month-1) and (features.region_month==features.region+'_'+features.month.astype(str)).all())
    freeze=json.loads((b/'frozen/selection.json').read_text());fm=json.loads((b/'frozen/configuration_manifest.json').read_text())
    check('Frozen selection bytes',sha(b/'frozen/selection.json')==fm['selection_sha256'])
    check('Frozen top5 order',freeze['configurations'][0]['features']==['wind_mean72_partial','sp_gust','sp_temperature','sp_ramp6','sp_mean24'])
    dictionary=pd.read_csv(a/'dictionaries/data_dictionary.csv')
    tablecols=sum(len(pd.read_parquet(a/n).columns) if n.endswith('.parquet') else len(pd.read_csv(a/n).columns) for n in ['regional_hourly.parquet','station_hourly.parquet']+[n+'.csv' for n in dims])
    check('Complete observational dictionary',len(dictionary)==tablecols and dictionary[['column','unit','meaning','source','transformation','temporal_convention','missing_representation','role','input_restriction']].notna().all().all())
    for sub in ['selection','evaluation']:
        membership=pd.read_parquet(b/sub/'split_membership.parquet')
        check(sub+' explicit keys',not membership.duplicated(['campaign','origin','role']+KEY).any())
        matched=membership.merge(base[KEY+['y_ENE_02','sp_wind']],on=KEY,how='left',validate='many_to_one')
        check(sub+' eligible memberships',matched[['y_ENE_02','sp_wind']].notna().all().all())
        for (origin,role),g in membership.groupby(['origin','role']):
            O=pd.Timestamp(origin+'-01',tz='America/Bahia');cut=O-pd.Timedelta(hours=169)
            if role in ['train','outer_train']:
                assert g.time_local.ge(cut-pd.DateOffset(months=3)).all() and g.time_local.lt(cut).all()
            elif role=='inner_fit':assert g.time_local.lt(O-pd.DateOffset(months=1)-pd.Timedelta(hours=169)).all()
            elif role=='inner_rank':assert g.time_local.ge(O-pd.DateOffset(months=1)).all() and g.time_local.lt(cut).all()
            else:
                end=min(O+pd.DateOffset(months=1),pd.Timestamp('2026-03-24 23:00',tz='America/Bahia')) if sub=='selection' else O+pd.DateOffset(months=1)
                assert g.time_local.ge(O).all() and g.time_local.lt(end).all()
        check(sub+' temporal availability',True)
    p=pd.read_parquet(b/'evaluation/predictions.parquet');ref=p[p.variant=='B'].sort_values(KEY)
    check('H2 size and prevalence',len(ref)==34811 and ref.y.sum()==11521 and ref.region.nunique()==8 and ref.origin.nunique()==6)
    target=base.set_index(KEY).y_ENE_02
    for v,g in p.groupby('variant'):
        g=g.sort_values(KEY);check(v+' paired keys/labels',g[KEY+['y','origin']].reset_index(drop=True).equals(ref[KEY+['y','origin']].reset_index(drop=True)))
        check(v+' labels equal Dataset 1',np.array_equal(g.y,target.loc[pd.MultiIndex.from_frame(g[KEY])].to_numpy()))
    calculated=metrics(data_root);central=pd.read_csv(b/'evaluation/main_table.csv').set_index('variant')
    for _,r in calculated[calculated.hypothesis=='H2'].iterrows():
        check(r.system+' exact metrics',all(abs(r[c]-central.loc[r.system,c])<1e-12 for c in ['n','positive','ap','brier','roc_auc','precision','recall','f1','specificity','alert_fraction']))
    # Every H1/H3/H4 outcome is independently matched to the observational target.
    hp=pd.read_parquet(b/'predictions/hypotheses.parquet')
    for (h,s),g in hp.groupby(['hypothesis','system']):
        check(h+' '+s+' unique population',not g.duplicated(KEY).any())
        reason=g.reason.iloc[0];y=base.set_index(KEY)['y_'+reason+'_02'].loc[pd.MultiIndex.from_frame(g[KEY])]
        check(h+' '+s+' observed labels',np.array_equal(g.y,y.to_numpy()))
    # Compare recalculated central metrics with the actual approved notebook tables.
    h1=pd.read_csv(b/'historical_reference/h1_calendarios_completo.csv')
    h3=pd.read_csv(b/'historical_reference/h3_metricas_completas.csv')
    h4=pd.read_csv(b/'historical_reference/h4_atualizacao_completo.csv')
    mappings={'H1':{'frequency28':0,'A_log':1,'A_multiscale':2},
              'H4':{'P0_logistic':0,'P1_access_only':1,'P2_access_only':2}}
    fields={'n':'n','positive':'positivos','ap':'AP','brier':'Brier','roc_auc':'ROC_AUC',
            'precision':'precisao','recall':'recall','f1':'F1','specificity':'especificidade'}
    for _,r in calculated[calculated.hypothesis!='H2'].iterrows():
        if r.hypothesis=='H3':
            reason,variant=r.system.split('_')
            reference=h3[(h3.Motivo==reason)&(h3['Informação']==('Calendário' if variant=='A' else 'Calendário + meteorologia'))].iloc[0]
        else:reference=({'H1':h1,'H4':h4}[r.hypothesis]).iloc[mappings[r.hypothesis][r.system]]
        check(r.hypothesis+' '+r.system+' approved notebook metrics',all(abs(r[k]-reference[v])<1e-12 for k,v in fields.items()))
    # Persisted explanations can be checked without rerunning a SHAP campaign.
    for split,n in [('evaluation',34811),('development',85185)]:
        obs=pd.read_parquet(b/'importance'/split/'observations.parquet')
        with np.load(b/'importance'/split/'shap_contributions.npz',allow_pickle=False) as z:phi=z['phi']
        check(split+' SHAP dimensions/order',phi.shape==(n,109) and np.array_equal(obs.row_id,np.arange(n)))
        check(split+' SHAP log-odds additivity',np.max(np.abs(expit(phi.sum(axis=1))-obs.p))<1e-12)
        flags=phi[:,57:108];vals=phi[:,6:57]
        check(split+' SHAP family accounting',np.allclose((vals+flags).sum(axis=1),phi[:,6:108].sum(axis=1),atol=1e-12))
    details['metrics']=calculated.to_dict('records')
    if models:
        import joblib,lightgbm
        check('LightGBM compatibility',lightgbm.__version__=='4.7.0')
        registry=json.loads((b/'models/model_registry.json').read_text());details['models']=[]
        check('Exactly 24 final models',len(registry)==24 and len({(r['origin'],r['variant']) for r in registry})==24)
        for r in registry:
            model_path=b/r['path'];assert sha(model_path)==r['sha256']
            model=joblib.load(model_path);g=p[(p.origin==r['origin'])&(p.variant==r['variant'])].copy()
            x=g[KEY].merge(base,on=KEY,how='left',validate='one_to_one');prep=model.named_steps['representation'];clf=model.named_steps['model']
            assert list(prep.numeric)==r['meteorological_columns'] and prep.cat_==CAL and list(clf.classes_)==[0,1]
            z=prep.transform(x);k=len(prep.numeric)
            assert z.shape==(len(g),6+2*k) and len(r['effective_columns'])==6+2*k
            assert np.array_equal(z[:,6:6+k],x[list(prep.numeric)].to_numpy(float),equal_nan=True)
            assert np.array_equal(z[:,6+k:],~np.isfinite(z[:,6:6+k]))
            prob=clf.booster_.predict(z,num_threads=4);delta=float(np.max(np.abs(prob-g.p.to_numpy())))
            check(r['origin']+' '+r['variant']+' historical inference',delta<1e-12)
            details['models'].append({'origin':r['origin'],'variant':r['variant'],'n':len(g),'max_probability_error':delta,'unknown_month_count':int(np.isnan(z[:,3]).sum()),'unknown_region_month_count':int(np.isnan(z[:,5]).sum())})
    return {'passed':all(c['passed'] for c in checks),'checks':checks,'n_checks':len(checks),'details':details}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['validate','metrics']);ap.add_argument('--data-root',required=True,type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--skip-models',action='store_true');a=ap.parse_args()
    if a.action=='metrics':
        result=metrics(a.data_root);text=result.to_csv(index=False)
    else:
        result=validate(a.data_root,not a.skip_models);text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if a.output:
        if a.output.exists():raise FileExistsError('Refusing to replace existing verification output')
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    if a.action=='metrics':print(text)
    else:print(json.dumps({'passed':result['passed'],'checks':result['n_checks']},indent=2))
if __name__=='__main__':main()
