"""Métricas externas e contrastes pareados com blocos de calendário sincronizados."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error
from .common import ROOT,P,DATA,OUT,REASONS,atomic_json,atomic_frame,stamp,note
from .experiments import metrics

KEYS=['time_local','region']
GROUP=['protocol','population','reason','variant','algorithm','seed']


class WeightedAP:
    """AP exata (inclusive empates) para reamostragem via pesos inteiros."""
    def __init__(self,y,p):
        y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
        self.order=np.argsort(-p,kind='stable');self.y=y[self.order]
        q=p[self.order];self.ends=np.r_[np.flatnonzero(np.diff(q)),len(q)-1]

    def __call__(self,weights):
        w=np.asarray(weights)[self.order]
        tp=np.cumsum(w*self.y)[self.ends];n=np.cumsum(w)[self.ends]
        if tp[-1]==0:return np.nan
        precision=np.divide(tp,n,out=np.zeros_like(tp,dtype=float),where=n>0)
        return np.sum(np.diff(np.r_[0.,tp])*precision)/tp[-1]


def synchronize(frames):
    merged=None
    for i,d in enumerate(frames):
        assert not d.duplicated(KEYS).any()
        piece=d[KEYS+['y','p_raw','p','threshold']].rename(columns={c:f'{c}_{i}' for c in ['y','p_raw','p','threshold']})
        merged=piece if merged is None else merged.merge(piece,on=KEYS,how='inner',validate='one_to_one')
    return merged.sort_values(KEYS).reset_index(drop=True)


def contrast(frames,coefficients,label,reason,population,block_hours=168,repetitions=500,same_target=True):
    d=synchronize(frames);assert len(d)>0
    if same_target:
        for i in range(1,len(frames)):np.testing.assert_array_equal(d.y_0,d[f'y_{i}'])
    hours=((d.time_local-stamp(P['final_origin'])).dt.total_seconds()/3600).to_numpy()
    blocks=np.floor(hours/block_hours).astype(int)
    # Keep even calendar blocks with no eligible rows; never draw regions independently.
    nblocks=int(np.ceil((stamp(P['final_end_exclusive'])-stamp(P['final_origin'])).total_seconds()/3600/block_hours))
    rng=np.random.default_rng(P['seed']+block_hours)
    draws=rng.multinomial(nblocks,np.full(nblocks,1/nblocks),size=repetitions)
    scorers=[WeightedAP(d[f'y_{i}'],d[f'p_raw_{i}']) for i in range(len(frames))]
    delta_loss=np.zeros(len(d));counts=np.bincount(blocks,minlength=nblocks)
    f1_counts=[]
    for i,c in enumerate(coefficients):
        y=d[f'y_{i}'].to_numpy();p=d[f'p_{i}'].to_numpy()
        delta_loss+=c*(p-y)**2
        predicted=p>=d[f'threshold_{i}'].to_numpy()
        f1_counts.append([np.bincount(blocks,weights=x,minlength=nblocks) for x in [y*predicted,(1-y)*predicted,y*(~predicted)]])
    loss_by_block=np.bincount(blocks,weights=delta_loss,minlength=nblocks)
    def f1(w):
        value=0.
        for c,(tp,fp,fn) in zip(coefficients,f1_counts):
            numerator=2*w@tp;denominator=numerator+w@fp+w@fn
            value+=c*(numerator/denominator if denominator else 0.)
        return value
    estimates={'ap_raw':sum(c*s(np.ones(len(d))) for c,s in zip(coefficients,scorers)),'brier':delta_loss.mean(),'f1':f1(np.ones(nblocks))}
    samples={'ap_raw':[],'brier':[],'f1':[]}
    for draw in draws:
        w=draw[blocks]
        samples['ap_raw'].append(sum(c*s(w) for c,s in zip(coefficients,scorers)))
        samples['brier'].append((draw@loss_by_block)/(draw@counts) if draw@counts else np.nan)
        samples['f1'].append(f1(draw))
    output=[]
    for metric,values in samples.items():
        a=np.asarray(values);finite=a[np.isfinite(a)];lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
        output.append(dict(contrast=label,reason=reason,population=population,n=len(d),positive_reference=int(d.y_0.sum()),metric=metric,estimate=estimates[metric],low=lo,high=hi,block_hours=block_hours,calendar_blocks=nblocks,nonempty_blocks=int((counts>0).sum()),replicates=repetitions,valid_replicates=len(finite),scope='conditional on fitted/selected models; retrospective; synchronized nonoverlapping calendar blocks'))
    return output


def group_metrics(d):
    if d.reason.iloc[0]=='magnitude':
        return dict(n=len(d),mae=mean_absolute_error(d.y,d.p_raw),rmse=np.sqrt(mean_squared_error(d.y,d.p_raw)),mean_y=d.y.mean(),mean_p=d.p_raw.mean())
    # Threshold may change at monthly origins, so a scalar threshold cannot score pooled P2.
    result=metrics(d.y,d.p,.5)
    result['ap_raw']=average_precision_score(d.y,d.p_raw) if d.y.sum() else np.nan
    result['brier_raw']=np.mean((d.y-d.p_raw)**2);result['mean_p_raw']=d.p_raw.mean()
    yh=d.p.to_numpy()>=d.threshold.to_numpy();y=d.y.to_numpy()
    tp=int(np.sum((y==1)&yh));fp=int(np.sum((y==0)&yh));fn=int(np.sum((y==1)&~yh));tn=int(np.sum((y==0)&~yh))
    result.update(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,threshold_min=d.threshold.min(),threshold_max=d.threshold.max())
    result.pop('threshold',None)
    return result


def summarize(predictions):
    rows=[]
    for scope,keys in [('all',[]),('month',['month']),('region',['region'])]:
        for key,d in predictions.groupby(GROUP+keys,observed=True):
            rows.append({**dict(zip(GROUP+keys,key)),'scope':scope,**group_metrics(d)})
    atomic_frame(pd.DataFrame(rows),OUT/'tables/final_metrics.csv')


def main_contrasts(predictions):
    p=predictions[predictions.seed.eq(P['seed'])];rows=[]
    def get(protocol,reason,variant,pop=None):
        z=p[(p.protocol==protocol)&(p.reason==reason)&(p.variant==variant)]
        return z if pop is None else z[z.population==pop]
    def add(frames,coef,label,reason,pop,same=True):
        for block in P['bootstrap_hours']:rows.extend(contrast(frames,coef,label,reason,pop,block,P['bootstrap_repetitions'],same))
        print('Contrast:',label,reason,pop,flush=True)
    for r in REASONS:
        for pop in ['anchor_complete','spatial_recent']:
            for protocol in ['P1','P2']:
                for plus,minus in [('B','A'),('D','C'),('C','A')]:
                    add([get(protocol,r,plus,pop),get(protocol,r,minus,pop)],[1,-1],f'{protocol} {plus}-{minus}',r,pop)
                for v in ['B','D']:
                    add([get(protocol,r,v,pop),get(protocol+'_frequency',r,'T',pop)],[1,-1],f'{protocol} {v}-frequency',r,pop)
            for v in ['A','B','C','D']:
                add([get('P2',r,v,pop),get('P1',r,v,pop)],[1,-1],f'P2-P1 {v}',r,pop)
            for v in ['A','B']:
                add([get('P1',r,v,pop),get('P0_logistic',r,v)],[1,-1],f'P1_system-P0 {v}',r,pop+' intersection V1')
            for protocol in ['P1_AB','P2_AB']:
                add([get(protocol,r,'B',pop),get(protocol,r,'A',pop)],[1,-1],f'{protocol} B-A',r,pop)
                add([get(protocol,r,'B',pop),get(protocol[:2]+'_frequency',r,'T',pop)],[1,-1],f'{protocol} B-frequency',r,pop)
            for v in ['A','B']:
                add([get('P2_AB',r,v,pop),get('P1_AB',r,v,pop)],[1,-1],f'P2_AB-P1_AB {v}',r,pop)
                add([get('P1_AB',r,v,pop),get('P0_logistic',r,v)],[1,-1],f'P1_AB_system-P0 {v}',r,pop+' intersection V1')
        for v in ['A','B']:
            add([get('P1_access_only',r,v),get('P0_logistic',r,v)],[1,-1],f'P1_access-P0 {v}',r,'v1_common')
            add([get('P2_access_only',r,v),get('P1_access_only',r,v)],[1,-1],f'P2_access-P1_access {v}',r,'v1_common')
        for protocol in ['P0_logistic','P1_access_only','P2_access_only']:
            add([get(protocol,r,'B'),get(protocol,r,'A')],[1,-1],f'{protocol} B-A',r,'v1_common')
        add([get('P1_access_only',r,'B'),get('P1_access_only',r,'A'),get('P0_logistic',r,'B'),get('P0_logistic',r,'A')],[1,-1,-1,1],'difference_in_differences update x MET',r,'v1_common')
        for latency in [24,720]:
            scenario=f'P1_latency{latency}'
            for plus,minus in [('B','A'),('D','C'),('C','A')]:
                add([get(scenario,r,plus,'spatial_recent'),get(scenario,r,minus,'spatial_recent')],[1,-1],f'{scenario} {plus}-{minus}',r,'spatial_recent')
            for v in ['A','D']:
                add([get(scenario,r,v,'spatial_recent'),get('P1',r,v,'spatial_recent')],[1,-1],f'{scenario}-P1 {v}',r,'spatial_recent')
    # H2: same hours and synchronized draws across reasons, while labels differ by reason.
    for protocol in ['P1','P2','P1_AB','P2_AB']:
        for pop in ['anchor_complete','spatial_recent']:
            for high,low in [('ENE','CNF'),('ENE','REL'),('CNF','REL')]:
                allframes=[get(protocol,r,v,pop) for r in REASONS for v in ['A','B']]
                common=synchronize(allframes)[KEYS]
                frames=[common.merge(get(protocol,r,v,pop),on=KEYS,validate='one_to_one') for r,v in [(high,'B'),(high,'A'),(low,'B'),(low,'A')]]
                add(frames,[1,-1,-1,1],f'H2 {protocol} METgain {high}-{low}',high+'-'+low,pop+' common reasons',False)
    for r in ['onset','end','widespread']:
        for plus,minus in [('B','A'),('D','C'),('C','A')]:
            add([get('P1_task',r,plus,'spatial_recent'),get('P1_task',r,minus,'spatial_recent')],[1,-1],f'P1_task {plus}-{minus}',r,'spatial_recent')
    # Same widespread target for both scores: tests whether target-specific training adds
    # ranking beyond using the regional occurrence score as a transfer baseline.
    for v in ['A','B','D']:
        task=get('P1_task','widespread',v,'spatial_recent')
        transfer=get('P1_AB' if v in ['A','B'] else 'P1','ENE',v,'spatial_recent').drop(columns='y')
        transfer=transfer.merge(task[KEYS+['y']],on=KEYS,validate='one_to_one')
        add([task,transfer],[1,-1],f'H6 widespread-specific minus ENE-score transfer {v}','widespread','spatial_recent common energy labels')
    for (protocol,r,pop),_ in p[p.protocol.isin(['P1_ensemble','P2_ensemble'])].groupby(['protocol','reason','population']):
        for plus,minus in [('B','A'),('D','C'),('C','A')]:
            add([get(protocol,r,plus,pop),get(protocol,r,minus,pop)],[1,-1],f'{protocol} {plus}-{minus}',r,pop)
        add([get(protocol,r,'B',pop),get(protocol[:2]+'_AB',r,'B',pop)],[1,-1],f'{protocol} B-best_AB',r,pop)
        add([get(protocol,r,'D',pop),get(protocol[:2],r,'D',pop)],[1,-1],f'{protocol} D-quad',r,pop)
    atomic_frame(pd.DataFrame(rows),OUT/'tables/paired_contrasts.csv')


def magnitude_contrasts(p):
    p=p[p.reason.eq('magnitude')&p.seed.eq(P['seed'])];rows=[]
    def get(protocol,v):return p[p.protocol.eq(protocol)&p.variant.eq(v)]
    comparisons=[('B-A',get('P1_task','B'),get('P1_task','A')),('D-C',get('P1_task','D'),get('P1_task','C')),('C-A',get('P1_task','C'),get('P1_task','A'))]
    comparisons += [(v+'-zero',get('P1_task',v),get('P1_task_zero','T')) for v in ['A','B','C','D']]
    for label,a,b in comparisons:
        d=synchronize([a,b]);np.testing.assert_allclose(d.y_0,d.y_1)
        losses=np.column_stack([abs(d.p_raw_0-d.y_0),abs(d.p_raw_1-d.y_1)])
        for hours in P['bootstrap_hours']:
            blocks=((d.time_local-stamp(P['final_origin'])).dt.total_seconds()/3600//hours).astype(int).to_numpy()
            nblocks=int(np.ceil((stamp(P['final_end_exclusive'])-stamp(P['final_origin'])).total_seconds()/3600/hours))
            rng=np.random.default_rng(P['seed']+hours);draws=rng.multinomial(nblocks,np.full(nblocks,1/nblocks),size=P['bootstrap_repetitions'])
            count=np.bincount(blocks,minlength=nblocks);delta=np.bincount(blocks,weights=losses[:,0]-losses[:,1],minlength=nblocks)
            values=(draws@delta)/(draws@count);lo,hi=np.quantile(values,[.025,.975])
            rows.append(dict(contrast=label,metric='MAE difference',n=len(d),estimate=np.mean(losses[:,0]-losses[:,1]),low=lo,high=hi,block_hours=hours,calendar_blocks=nblocks))
    atomic_frame(pd.DataFrame(rows),OUT/'tables/magnitude_contrasts.csv')


def diagnostics(predictions):
    b=pd.read_parquet(DATA/'analytical.parquet');p=predictions[predictions.seed.eq(P['seed'])]
    selected=p[p.protocol.isin(['P1','P2','P1_AB','P2_AB','P1_task'])]
    b=b.sort_values(['region','time_local']).copy()
    # Only post-hoc diagnostics use the true immediately preceding state/episode age.
    age=np.zeros(len(b),dtype=int);last_region=None;last_y=np.nan;elapsed=0;known_start=False
    for i,row in enumerate(b[['region','y_ENE_02']].itertuples(index=False)):
        if row.region!=last_region:last_y=np.nan;elapsed=0;known_start=False
        if row.y_ENE_02==1:
            if last_y==0:elapsed=1;known_start=True
            elif known_start:elapsed+=1
        else:elapsed=0;known_start=False
        age[i]=elapsed if known_start else 0;last_region=row.region;last_y=row.y_ENE_02
    b['episode_age']=age
    columns=KEYS+['reference_mwh','previous_ENE','sp_count','sp_distance','wind_lag1','episode_age']+[f'y_{r}_{tau:02}' for r in REASONS for tau in [1,5]]
    # Feature names are asserted against the actual schema, not inferred from prose.
    columns=[c for c in columns if c in b]
    joined=selected.merge(b[columns],on=KEYS,how='left',validate='many_to_one');rows=[]
    for key,d in joined.groupby(GROUP,observed=True):
        info=dict(zip(GROUP,key))
        for name,mask in [('reference_ge10',d.reference_mwh.ge(10)),('anchor_wind_missing',d.wind_lag1.isna()),('anchor_wind_observed',d.wind_lag1.notna()),('distance_le100',d.sp_distance.le(100)),('distance_gt100',d.sp_distance.gt(100)),('sensor_count_le2',d.sp_count.le(2)),('sensor_count_ge3',d.sp_count.ge(3))]:
            if mask.any():rows.append({**info,'diagnostic':name,**group_metrics(d[mask])})
        r=info['reason']
        for tau in [1,5]:
            col=f'y_{r}_{tau:02}'
            if col not in d:continue
            z=d[d[col].notna()].copy();z['y']=z[col]
            if len(z):rows.append({**info,'diagnostic':f'relabel_tau{tau}_fixed_probabilities_not_retrained',**group_metrics(z)})
        if r=='ENE':
            for prev in [0,1]:
                z=d[d.previous_ENE.eq(prev)]
                if len(z):rows.append({**info,'diagnostic':f'previous_state_{prev}_oracle_error_stratum_not_input',**group_metrics(z)})
            for name,mask in [('episode_age1',d.episode_age.eq(1)),('episode_age2_4',d.episode_age.between(2,4)),('episode_age5plus',d.episode_age.ge(5)),('episode_left_censored',d.episode_age.eq(0)&d.y.eq(1))]:
                z=d[mask]
                if len(z):
                    values=group_metrics(z);values['ap']=values['ap_raw']=np.nan
                    rows.append({**info,'diagnostic':name+'_positive_only_recall_not_ranking',**values})
        if r in ['onset','end']:
            z=d[d.previous_ENE.eq(0 if r=='onset' else 1)]
            if len(z):rows.append({**info,'diagnostic':'oracle_at_risk_state_known_for_evaluation_not_model_input',**group_metrics(z)})
        if r=='magnitude':
            for name,mask in [('positive_magnitude',d.y.gt(0)),('zero_magnitude',d.y.eq(0))]:
                z=d[mask]
                if len(z):rows.append({**info,'diagnostic':name,**group_metrics(z)})
    atomic_frame(pd.DataFrame(rows),OUT/'tables/robustness_strata.csv')
    # Direct recalibration/refit compatibility, raw AP and probability metrics on each block.
    compat=p[p.protocol.isin(['P0_logistic','P0_intercept','P1','P1_holdcal','P1_AB','P1_AB_holdcal'])]
    rows=[]
    for key,d in compat.groupby(GROUP,observed=True):rows.append({**dict(zip(GROUP,key)),**group_metrics(d)})
    atomic_frame(pd.DataFrame(rows),OUT/'tables/calibration_external.csv')
    atomic_frame(pd.DataFrame([dict(zip(GROUP,key))|group_metrics(d) for key,d in predictions[predictions.protocol.isin(['P1','P1_AB','P1_task'])].groupby(GROUP,observed=True)]),OUT/'tables/seed_external.csv')


def main():
    p=pd.read_parquet(DATA/'final_predictions.parquet');p['month']=p.time_local.dt.strftime('%Y-%m')
    assert not p.duplicated(['protocol','population','reason','variant','seed']+KEYS).any()
    assert p.time_local.ge(stamp(P['final_origin'])).all() and p.time_local.lt(stamp(P['final_end_exclusive'])).all()
    summarize(p);diagnostics(p);main_contrasts(p);magnitude_contrasts(p)
    atomic_json(OUT/'tables/evaluation_semantics.json',dict(primary='AP raw; B-A/D-C within identical rows',bootstrap='500 nonoverlapping calendar block resamples; synchronized regions; fixed models; 168/672h',selection_uncertainty_included=False,historically_blind=False,probability_score_direction='lower Brier/log loss is better',thresholds='selected in earlier OOF; pooled F1 respects varying monthly thresholds',caution='many exploratory contrasts, no multiplicity correction; 672h has only seven calendar blocks'))
    note('R4 avaliação','Métricas gerais/mensais/regionais, sensibilidade de alvos e população, contrastes pareados AP/Brier/F1 e H2 com blocos sincronizados concluídos. Intervalos condicionais, sem incerteza de seleção; resultados retrospectivos.')


if __name__=='__main__':main()
