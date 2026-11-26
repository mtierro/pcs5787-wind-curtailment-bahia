"""Temporal, frozen-list and execution checks; no classifier training."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score
from audit_eval import HERE,DEV,ROOT,sha,registered,verify_frozen,digest
from eval_core import (FEATURES,CALENDAR,REDUCED,SPECS,VARIANTS,KEYS,ORIGINS,START,STOP,
                       columns,spec,cutoff,split,load_base,load_reference,align,threshold,warm_start,
                       matrix,score,fingerprint,WeightedAP)


def read(name):return json.loads((HERE/name).read_text())


@pytest.fixture(scope='module')
def base():return load_base()


def test_frozen_integrity_and_exact_feature_order():
    registered();frozen=verify_frozen()
    assert columns('C_top5')==['wind_mean72_partial','sp_gust','sp_temperature','sp_ramp6','sp_mean24']
    assert columns('C_spatial')==FEATURES[-14:]
    for v in REDUCED:
        c=columns(v);assert c==[f for f in FEATURES if f in c]
        assert SPECS[v]['effective_columns']==CALENDAR+c+['missing::'+f for f in c]
    with pytest.raises(AssertionError):spec('A')
    with pytest.raises(AssertionError):spec('B')


def test_exact_clock_populations_and_historical_memberships(base):
    membership=pd.read_parquet(HERE/'split_membership.parquet')
    checks=pd.read_csv(HERE/'historical_reproduction.csv')
    for origin in ORIGINS:
        tr,va=split(base,origin);end=origin-pd.Timedelta(hours=169);start=end-pd.DateOffset(months=3)
        eligible=base.y_ENE_02.notna()&base.sp_wind.notna()
        expected=base.loc[eligible&base.time_local.ge(start)&base.time_local.lt(end)]
        assert np.array_equal(tr.row_id,expected.row_id)
        assert tr.time_local.max()+pd.Timedelta(hours=169)<origin
        for role,frame in [('train',tr),('evaluation',va)]:
            ids=membership.loc[membership.origin.eq(origin.strftime('%Y-%m'))&membership.role.eq(role),'row_id']
            assert np.array_equal(frame.row_id,ids)
        rec=checks[checks.origin.eq(origin.strftime('%Y-%m'))]
        assert rec.train_population_sha256.eq(fingerprint(tr)).all()
        assert rec.eval_population_sha256.eq(fingerprint(va)).all()
    assert checks.n.sum()==69622 and checks.positive.sum()==23042 and checks.residual_max.eq(0).all()


def test_strict_population_alignment_rejects_loss_and_relabeling():
    ref=load_reference();align(ref,['A','B'])
    with pytest.raises(AssertionError):align(ref.drop(index=ref.index[0]),['A','B'])
    changed=ref.copy();changed.loc[changed.index[0],'y']=1-changed.iloc[0].y
    with pytest.raises(AssertionError):align(changed,['A','B'])


def test_threshold_admissibility_poisoned_future_labels():
    origin=ORIGINS[1];end=cutoff(origin)
    times=pd.date_range(end-pd.Timedelta(hours=250),periods=250,freq='h')
    pool=pd.DataFrame(dict(time_local=times,region=['r']*250,y=np.tile([0,1],125),p=np.tile([.1,.9],125)))
    t,record=threshold(pool,origin)
    bad=pd.DataFrame(dict(time_local=[end,origin,origin+pd.Timedelta(hours=1)],region=['r']*3,y=[1,1,0],p=[.001,.002,.999]))
    other,rec=threshold(pd.concat([pool,bad]),origin)
    assert t==other and rec==record and record['n']==250
    for v in REDUCED:
        h,_=warm_start(v);t,_=threshold(h,START)
        expected=next(r['threshold'] for r in read_frozen()['initial_april_thresholds'] if r['variant']==v)
        assert t==expected


def read_frozen():return json.loads((DEV/'frozen/selection.json').read_text())


def test_weighted_ap_ties_matches_sklearn():
    y=np.array([0,1,1,0,1,0]);p=np.array([.4,.4,.8,.1,.8,.9]);w=np.array([2,1,0,3,2,0])
    assert abs(WeightedAP(y,p)(w)-average_precision_score(y,p,sample_weight=w))<1e-14


@pytest.fixture(scope='module')
def results():
    if not (HERE/'predictions.parquet').exists():pytest.skip('Run reduced campaign first')
    return align(pd.read_parquet(HERE/'predictions.parquet'))


def test_saved_reduced_models_probabilities_flags_and_fixed_lists(base,results):
    rows=read('reduced_models.json');assert len(rows)==12
    for r in rows:
        v=r['variant'];origin=pd.Timestamp(r['origin']+'-01',tz='America/Bahia');tr,va=split(base,origin)
        assert r['features']==columns(v) and r['feature_list_sha256']==SPECS[v]['feature_list_sha256']
        assert r['missing_indicators']==['missing::'+c for c in columns(v)]
        assert r['train_population_sha256']==fingerprint(tr) and r['eval_population_sha256']==fingerprint(va)
        assert sha(HERE/r['model_path'])==r['model_sha256']
        model=joblib.load(HERE/r['model_path']);z=matrix(model,va,v)
        assert np.isnan(z[:,3]).all() and np.isnan(z[:,5]).all()
        p=model.named_steps['model'].booster_.predict(z,num_threads=4)
        saved=results[v].loc[results[v].origin.eq(r['origin'])]
        assert np.array_equal(p,saved.p.to_numpy())
    repeated=[r for r in rows if 'repeat_probability_max_residual' in r]
    assert len(repeated)==2 and all(r['repeat_probability_max_residual']==0 for r in repeated)
    assert read('provenance/training.json')['baseline_refits']==0


def test_threshold_reconstruction_all_variants(results):
    for v in VARIANTS:
        pool,_=warm_start(v)
        for origin in ORIGINS:
            t,record=threshold(pool,origin)
            sub=results[v].loc[results[v].origin.eq(origin.strftime('%Y-%m'))]
            assert abs(t-sub.threshold.iloc[0])<1e-14
            pool=pd.concat([pool,sub[KEYS+['y','p']]],ignore_index=True)


def test_metric_accounting_and_absolute_vs_monthly_AP(results):
    if not (HERE/'summary.csv').exists():pytest.skip('Run analysis first')
    summary=pd.read_csv(HERE/'summary.csv').set_index('variant')
    monthly=pd.read_csv(HERE/'monthly_metrics.csv')
    for v in VARIANTS:
        f=results[v];m=score(f.y,f.p,f.threshold);record=summary.loc[v]
        for key,value in m.items():assert abs(value-record[key])<1e-12,key
        g=monthly[monthly.variant.eq(v)]
        for key in ['n','positive','tp','fp','fn','tn','alert_region_hours']:assert g[key].sum()==record[key]
        assert abs(np.average(g.brier,weights=g.n)-record.brier)<1e-14
        assert abs(g.ap.mean()-record.mean_monthly_ap)<1e-14
        assert abs(g.ap.mean()-record.ap)>1e-6


def test_bootstrap_synchronized_draws_and_independent_metrics(results):
    if not (HERE/'bootstrap_replicates.parquet').exists():pytest.skip('Run bootstrap first')
    samples=pd.read_parquet(HERE/'bootstrap_replicates.parquet');ref=results['B'];y=ref.y.to_numpy(int)
    for d in read('bootstrap_diagnostics.json'):
        nb=d['blocks'];hours=d['block_hours']
        draws=np.random.default_rng(d['seed']).multinomial(nb,np.full(nb,1/nb),size=2000)
        assert hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()==d['draw_sha256']
        blocks=((ref.time_local-START).dt.total_seconds()/3600//hours).to_numpy(int)
        w=draws[0,blocks]
        for v in VARIANTS:
            row=samples.loc[samples.variant.eq(v)&samples.block_hours.eq(hours)&samples.rep.eq(0)].iloc[0]
            assert abs(average_precision_score(y,results[v].p,sample_weight=w)-row.ap)<1e-12
            assert abs(np.average((results[v].p-y)**2,weights=w)-row.brier)<1e-12
    assert samples.groupby(['variant','block_hours']).size().eq(2000).all()
    intervals=pd.read_csv(HERE/'paired_intervals.csv')
    assert set(intervals.contrast)=={'C_top5_minus_B','C_spatial_minus_B','B_minus_A','C_top5_minus_A','C_spatial_minus_A'}
    for v in REDUCED:
        rows=intervals[intervals.contrast.eq(v+'_minus_B')]
        passed=bool((rows.ap_low>-.01).all() and (rows.ap_simultaneous_low>-.01).all())
        assert read('conclusions.json')[v]['noninferior_under_registered_rule']==passed


def test_preservation_of_all_historical_files():
    from audit_eval import verify,verify_prior_manifest
    assert verify(write=False)['unchanged']
    for folder in ['reports/h2_feature_importance','reports/h2_feature_selection_dev']:
        verify_prior_manifest(folder)
