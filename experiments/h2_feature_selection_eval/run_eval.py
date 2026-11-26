"""Historical reproduction gate, then twelve frozen reduced fits; no reselection."""
import argparse
from datetime import datetime,timezone
import json
import resource
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from audit_eval import HERE,ROOT,DEV,sha,digest,save,write_json,registered
from eval_core import (CONFIG,FROZEN,FEATURES,CALENDAR,SPECS,REDUCED,KEYS,ORIGINS,START,STOP,FINAL_IDS,
                       load_base,split,load_reference,warm_start,threshold,matrix,score,align,columns,spec,fingerprint,make_model,stamp,cutoff)


def preflight():
    reg=registered();start=time.monotonic()
    assert reg['dependencies']['lightgbm']=='4.7.0'
    base=load_base();ref=load_reference();aligned=align(ref,['A','B'])
    checks=[];ledgers=[];thresholds=[];members=[];dates=[];input_paths=[]
    composition=pd.read_csv(ROOT/'reports/v2/tables/final_training_composition.csv')
    calibration=pd.read_csv(ROOT/'reports/v2/tables/final_calibration_history.csv')
    for i,origin in enumerate(ORIGINS):
        tr,va=split(base,origin);label=origin.strftime('%Y-%m')
        for role,frame in [('train',tr),('evaluation',va)]:
            members.append(pd.DataFrame(dict(origin=label,role=role,row_id=frame.row_id.to_numpy())))
            dates.append(dict(origin=label,role=role,n=len(frame),positive=int(frame.y_ENE_02.sum()),
                              first=str(frame.time_local.min()),last=str(frame.time_local.max()),population_sha256=fingerprint(frame)))
        comp=composition.loc[composition.protocol.eq('P2')&composition.reason.eq('ENE')&composition.population.eq('spatial_recent')&composition.origin.eq(origin.strftime('%Y-%m-%d'))]
        for year,g in tr.groupby(tr.time_local.dt.year):
            row=comp.loc[comp.year.eq(year)];assert len(row)==1
            assert int(row.iloc[0].n)==len(g)
            assert row.iloc[0].keys_sha256==digest(pd.util.hash_pandas_object(g[['region','time_local']],index=False).tolist())
        for v,ident in zip(['A','B'],FINAL_IDS[i]):
            mp=ROOT/f'models/v2/{ident}.joblib';jp=ROOT/f'.scratch/v2/cache/{ident}.json';pp=jp.with_suffix('.parquet')
            input_paths.extend([mp,jp,pp])
            meta=json.loads(jp.read_text());assert sha(mp)==meta['model_sha256'] and sha(pp)==meta['pred_sha256']
            assert meta['columns']==columns(v) and meta['n_train']==len(tr) and meta['positive_train']==tr.y_ENE_02.sum()
            assert stamp(meta['train_start'])==tr.time_local.min() and stamp(meta['train_end'])==tr.time_local.max()
            frame=aligned[v].loc[aligned[v].origin.eq(label)].reset_index(drop=True)
            assert set(frame.run_id)=={ident} and len(frame)==len(va)
            pd.testing.assert_frame_equal(frame[KEYS],va[KEYS].reset_index(drop=True),check_dtype=False)
            assert np.array_equal(frame.y,va.y_ENE_02)
            cache=pd.read_parquet(pp).sort_values(KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(cache[KEYS+['y']],frame[KEYS+['y']],check_dtype=False)
            assert np.array_equal(cache.p,frame.p)
            model=joblib.load(mp);z=matrix(model,va,v)
            with threadpool_limits(limits=4):p=model.predict_proba(va)[:,1]
            native=model.named_steps['model'].booster_.predict(z,num_threads=4)
            residual=float(np.max(np.abs(p-frame.p.to_numpy())))
            assert residual<=CONFIG['tolerances']['probability'] and np.array_equal(p,native)
            row=calibration.loc[calibration.protocol.eq('P2')&calibration.reason.eq('ENE')&calibration.population.eq('spatial_recent')&calibration.variant.eq(v)&calibration.origin.eq(origin.strftime('%Y-%m-%d'))]
            assert len(row)==1 and row.iloc[0]['method']=='none'
            assert frame.threshold.nunique()==1 and abs(frame.threshold.iloc[0]-row.iloc[0].threshold)<1e-14
            checks.append(dict(origin=label,variant=v,run_id=ident,n=len(frame),positive=int(frame.y.sum()),n_train=len(tr),
                               residual_max=residual,threshold=float(frame.threshold.iloc[0]),n_entries=z.shape[1],
                               month_unknown=int(np.isnan(z[:,3]).sum()),region_month_unknown=int(np.isnan(z[:,5]).sum()),
                               model_sha256=sha(mp),train_population_sha256=fingerprint(tr),eval_population_sha256=fingerprint(va)))
            ledgers.append(dict(origin=label,variant=v,model_path=str(mp.relative_to(ROOT)),model_sha256=sha(mp),
                                metadata=meta,parameters=model.named_steps['model'].get_params()))
    for v in ['A','B']:
        pool,source=warm_start(v)
        for origin in ORIGINS:
            label=origin.strftime('%Y-%m');frame=aligned[v].loc[aligned[v].origin.eq(label)]
            t,record=threshold(pool,origin)
            assert abs(t-frame.threshold.iloc[0])<1e-14
            thresholds.append(dict(variant=v,origin=label,threshold=t,source=source,**record))
            pool=pd.concat([pool,frame[KEYS+['y','p']]],ignore_index=True)
    old=pd.read_csv(ROOT/'reports/v2/tables/final_metrics.csv')
    old=old.loc[old.protocol.eq('P2')&old.reason.eq('ENE')&old.population.eq('spatial_recent')&old.algorithm.eq('lightgbm')&old.seed.eq(5787)&old.variant.isin(['A','B'])]
    metric_checks=[]
    for scope,key in [('all',None),('month','origin'),('region','region')]:
        for v,frame in aligned.items():
            groups=[('all',frame)] if key is None else list(frame.groupby(key))
            for name,g in groups:
                m=score(g.y,g.p,g.threshold);row=old.loc[old.scope.eq(scope)&old.variant.eq(v)]
                if scope!='all':row=row.loc[row['month' if scope=='month' else 'region'].eq(name)]
                assert len(row)==1
                for metric in ['n','positive','prevalence','ap','brier','roc_auc','precision','recall','f1','tp','fp','fn','tn']:
                    residual=abs(m[metric]-row.iloc[0][metric]);assert residual<=1e-12,(scope,v,name,metric,residual)
                metric_checks.append(dict(scope=scope,stratum=name,variant=v,**m))
    input_paths.extend([ROOT/n for n in ['data/processed/v2/analytical.parquet','data/processed/v2/final_predictions.parquet','data/processed/v2/selected_oof.parquet','reports/v2/tables/final_metrics.csv','reports/v2/tables/final_calibration_history.csv','reports/v2/tables/final_training_composition.csv','configs/v2/finalists_frozen.json','configs/v2/feature_groups.json','configs/v2/protocol.json','src/pcs5787/v2/learners.py','src/pcs5787/v2/common.py','src/pcs5787/v2/calibration.py','src/pcs5787/v2/experiments.py','src/pcs5787/v2/final_campaign.py','src/pcs5787/v2/evaluation.py']])
    input_paths.extend([DEV/'frozen/selection.json',DEV/'frozen/configuration_manifest.json',DEV/'predictions.parquet'])
    save(pd.DataFrame([dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(input_paths))]),'provenance/inputs.csv')
    save(pd.DataFrame(checks),'historical_reproduction.csv');save(pd.DataFrame(metric_checks),'historical_metrics_checks.csv')
    save(pd.DataFrame(thresholds),'historical_threshold_checks.csv');write_json(HERE/'historical_models.json',ledgers)
    save(base[['row_id']+KEYS+['y_ENE_02','sp_wind']],'population.parquet')
    save(pd.concat(members,ignore_index=True),'split_membership.parquet');save(pd.DataFrame(dates),'split_dates.csv')
    save(ref[KEYS+['y','p','threshold','origin','variant','run_id']],'historical_predictions.parquet')
    write_json(HERE/'provenance/preflight.json',dict(utc=datetime.now(timezone.utc).isoformat(),passed=True,
        n_models=12,n_per_variant=34811,positive=11521,region_count=8,probability_residual_max=max(c['residual_max'] for c in checks),
        historical_thresholds_reconstructed=True,all_historical_metric_strata_matched=True,
        seconds=time.monotonic()-start,inputs_manifest_sha256=sha(HERE/'provenance/inputs.csv'),
        source_sha256={n:sha(HERE/n) for n in ['eval_core.py','run_eval.py','audit_eval.py']},protocol_sha256=sha(HERE/'protocol.json')))
    print('PASS: 12 historical models, 34,811 rows each; probabilities, thresholds and all metric strata reproduced',flush=True)


def run():
    registered()
    if (HERE/'predictions.parquet').exists():raise RuntimeError('Outputs exist; use a new package for reproduction')
    gate=json.loads((HERE/'provenance/preflight.json').read_text());assert gate['passed'] and gate['n_models']==12
    assert sha(HERE/'provenance/inputs.csv')==gate['inputs_manifest_sha256']
    for row in pd.read_csv(HERE/'provenance/inputs.csv').itertuples():assert sha(ROOT/row.path)==row.sha256
    start=time.monotonic();base=load_base();preds=[pd.read_parquet(HERE/'historical_predictions.parquet')]
    ledgers=[];thresholds=[];members=[];warm_rows=[]
    pools={};sources={}
    for v in REDUCED:
        pools[v],sources[v]=warm_start(v)
        warm_rows.append(pools[v].assign(variant=v))
    save(pd.concat(warm_rows,ignore_index=True),'reduced_warm_start.parquet')
    for origin in ORIGINS:
        tr,va=split(base,origin);label=origin.strftime('%Y-%m')
        for v in REDUCED:
            cfg=SPECS[v];cols=columns(v);t,history=threshold(pools[v],origin)
            if origin==START:
                frozen_t=next(r['threshold'] for r in FROZEN['initial_april_thresholds'] if r['variant']==v)
                assert t==frozen_t
            thresholds.append(dict(variant=v,origin=label,threshold=t,source=sources[v],**history))
            h=pools[v].loc[pools[v].time_local.ge(cutoff(origin)-pd.Timedelta(days=90))&pools[v].time_local.lt(cutoff(origin))]
            members.append(h[KEYS+['y','p']].assign(variant=v,origin=label))
            model=make_model(spec(v),cols);fitstart=time.perf_counter()
            with threadpool_limits(limits=4),warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(tr,tr.y_ENE_02.to_numpy(int),model__categorical_feature=list(range(6)))
            fit_seconds=time.perf_counter()-fitstart
            predstart=time.perf_counter();z=matrix(model,va,v)
            p=model.named_steps['model'].booster_.predict(z,num_threads=4);predict_seconds=time.perf_counter()-predstart
            assert np.isfinite(p).all() and z.shape[1]==len(cfg['effective_columns'])
            out=va[['row_id']+KEYS].copy();out['y']=va.y_ENE_02.to_numpy(int);out['p']=p;out['threshold']=t
            out['variant']=v;out['origin']=label;out['run_id']=f'{v}_{label}'
            path=HERE/f'models/{label}/{v}.joblib';path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,path,compress=3)
            record=dict(variant=v,origin=label,features=cols,missing_indicators=cfg['missing_indicators'],
                        effective_columns=cfg['effective_columns'],feature_list_sha256=digest(cols),
                        n_train=len(tr),positive_train=int(tr.y_ENE_02.sum()),n_eval=len(va),
                        train_first=str(tr.time_local.min()),train_last=str(tr.time_local.max()),
                        train_population_sha256=fingerprint(tr),eval_population_sha256=fingerprint(va),
                        last_training_label_available=str(tr.time_local.max()+pd.Timedelta(hours=169)),
                        fit_seconds=fit_seconds,predict_seconds=predict_seconds,threshold=t,threshold_history=history,
                        model_path=str(path.relative_to(HERE)),model_sha256=sha(path),parameters=model.named_steps['model'].get_params(),
                        booster_parameters=model.named_steps['model'].booster_.params,
                        encoder_categories=[a.tolist() for a in model.named_steps['representation'].encoder_.categories_],
                        month_unknown=int(np.isnan(z[:,3]).sum()),region_month_unknown=int(np.isnan(z[:,5]).sum()),
                        warnings=sorted(set(str(w.message) for w in caught)))
            if origin==START:
                repeated=make_model(spec(v),cols)
                with threadpool_limits(limits=4):repeated.fit(tr,tr.y_ENE_02.to_numpy(int),model__categorical_feature=list(range(6)))
                repeated_p=repeated.named_steps['model'].booster_.predict(matrix(repeated,va,v),num_threads=4)
                assert np.array_equal(repeated_p,p);record['repeat_probability_max_residual']=float(np.max(np.abs(repeated_p-p)))
            ledgers.append(record);preds.append(out)
            pools[v]=pd.concat([pools[v],out[KEYS+['y','p']]],ignore_index=True)
        print(label,'reduced models completed;',len(tr),'train,',len(va),'evaluation',flush=True)
    combined=pd.concat(preds,ignore_index=True);align(combined)
    save(combined,'predictions.parquet');save(pd.DataFrame(thresholds),'reduced_threshold_history.csv')
    save(pd.concat(members,ignore_index=True),'threshold_membership.parquet');write_json(HERE/'reduced_models.json',ledgers)
    write_json(HERE/'provenance/training.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        models_saved=12,determinism_repeat_fits=2,baseline_refits=0,seconds=time.monotonic()-start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        source_sha256={n:sha(HERE/n) for n in ['eval_core.py','run_eval.py','audit_eval.py']},
        protocol_sha256=sha(HERE/'protocol.json'),selection_sha256=sha(DEV/'frozen/selection.json'),
        selected_features_updated=False,final_evaluation_executed=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','run'])
    if parser.parse_args().stage=='preflight':preflight()
    else:run()
