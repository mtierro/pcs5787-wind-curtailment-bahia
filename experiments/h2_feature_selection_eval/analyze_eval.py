"""Pre-registered paired temporal bootstrap and descriptive final evaluation."""
from datetime import datetime,timezone
import hashlib
import json
import resource
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from audit_eval import HERE,sha,save,write_json,registered
from eval_core import CONFIG,VARIANTS,REDUCED,START,STOP,columns,align,score,WeightedAP


def summaries(aligned):
    pooled=[];monthly=[];regional=[];cells=[]
    for v,f in aligned.items():
        k=len(columns(v))
        m=score(f.y,f.p,f.threshold)
        rows=[]
        for origin,g in f.groupby('origin',sort=True):
            row=dict(variant=v,origin=origin,threshold=float(g.threshold.iloc[0]),**score(g.y,g.p,g.threshold))
            rows.append(row);monthly.append(row)
        pooled.append(dict(variant=v,n_meteorological=k,n_effective=6+2*k,**m,
                           mean_monthly_ap=float(np.mean([r['ap'] for r in rows])),
                           min_monthly_ap=float(min(r['ap'] for r in rows)),max_monthly_ap=float(max(r['ap'] for r in rows))))
        for region,g in f.groupby('region',sort=True):regional.append(dict(variant=v,region=region,**score(g.y,g.p,g.threshold)))
        for (origin,region),g in f.groupby(['origin','region'],sort=True):
            cells.append(dict(variant=v,origin=origin,region=region,**score(g.y,g.p,g.threshold)))
    return pd.DataFrame(pooled),pd.DataFrame(monthly),pd.DataFrame(regional),pd.DataFrame(cells)


def contrasts_table(table,keys):
    metrics=['ap','brier','roc_auc','precision','recall','f1','specificity','alert_fraction']
    rows=[]
    indexed=table.set_index(['variant']+keys)
    for contrast in CONFIG['contrasts']:
        left=indexed.loc[contrast['left']];right=indexed.loc[contrast['right']]
        assert left.index.equals(right.index) and np.array_equal(left.n,right.n) and np.array_equal(left.positive,right.positive)
        for key,row in left.iterrows():
            reference=right.loc[key];values=key if isinstance(key,tuple) else (key,)
            rows.append(dict(contrast=contrast['name'],**dict(zip(keys,values)),n=int(row.n),
                             positive=int(row.positive),prevalence=float(row.prevalence),
                             **{'delta_'+m:float(row[m]-reference[m]) for m in metrics}))
    return pd.DataFrame(rows)


def bootstrap(aligned):
    ref=aligned['B'];y=ref.y.to_numpy(int);n=len(y)
    scorers={v:WeightedAP(y,aligned[v].p.to_numpy()) for v in VARIANTS}
    estimates={v:score(y,aligned[v].p,aligned[v].threshold) for v in VARIANTS}
    abs_samples=[];diff_samples=[];absolute_intervals=[];paired_intervals=[];diagnostics=[];block_rows=[]
    for hours in CONFIG['bootstrap']['block_hours']:
        nb=int(np.ceil((STOP-START)/pd.Timedelta(hours=hours)))
        block=((ref.time_local-START).dt.total_seconds().to_numpy()/3600//hours).astype(int)
        counts=np.bincount(block,minlength=nb)
        positives=np.bincount(block,weights=y,minlength=nb)
        for i in range(nb):
            block_rows.append(dict(block_hours=hours,block=i,start=str(START+pd.Timedelta(hours=hours*i)),
                                   stop_exclusive=str(min(START+pd.Timedelta(hours=hours*(i+1)),STOP)),
                                   n=int(counts[i]),positive=int(positives[i])))
        seed=CONFIG['bootstrap']['seeds'][str(hours)]
        draws=np.random.default_rng(seed).multinomial(nb,np.full(nb,1/nb),size=CONFIG['bootstrap']['replicates'])
        losses={v:np.bincount(block,weights=(aligned[v].p.to_numpy()-y)**2,minlength=nb) for v in VARIANTS}
        confusion={}
        for v in VARIANTS:
            pred=aligned[v].p.to_numpy()>=aligned[v].threshold.to_numpy()
            confusion[v]=[np.bincount(block,weights=a,minlength=nb) for a in [(y==1)&pred,(y==0)&pred,(y==1)&~pred]]
        arrays={v:np.zeros((len(draws),3)) for v in VARIANTS}
        for rep,draw in enumerate(draws):
            weights=draw[block];den=float(draw@counts)
            for v in VARIANTS:
                tp,fp,fn=[float(draw@a) for a in confusion[v]];f1den=2*tp+fp+fn
                arrays[v][rep]=[scorers[v](weights),float(draw@losses[v])/den,2*tp/f1den if f1den else 0.]
        assert all(np.isfinite(a).all() for a in arrays.values())
        observed=[];diff=[]
        for c in CONFIG['contrasts']:
            left,right=c['left'],c['right']
            observed.append(estimates[left]['ap']-estimates[right]['ap'])
            diff.append(arrays[left]-arrays[right])
        primary=[i for i,c in enumerate(CONFIG['contrasts']) if c['primary_family']]
        sd=np.array([d[:,0].std(ddof=1) for d in diff])
        centered=np.column_stack([diff[i][:,0]-observed[i] for i in primary])
        standardized=np.divide(centered,sd[primary],out=np.zeros_like(centered),where=sd[primary]>0)
        critical=float(np.quantile(standardized.max(axis=1),.975))
        for v in VARIANTS:
            a=arrays[v]
            abs_samples.append(pd.DataFrame(dict(variant=v,block_hours=hours,rep=np.arange(len(draws)),ap=a[:,0],brier=a[:,1],f1=a[:,2])))
            row=dict(variant=v,block_hours=hours,replicates=len(draws),seed=seed)
            for j,m in enumerate(['ap','brier','f1']):
                row.update({m+'_estimate':estimates[v][m],m+'_low':float(np.quantile(a[:,j],.025)),m+'_high':float(np.quantile(a[:,j],.975))})
            absolute_intervals.append(row)
        for i,c in enumerate(CONFIG['contrasts']):
            d=diff[i]
            diff_samples.append(pd.DataFrame(dict(contrast=c['name'],block_hours=hours,rep=np.arange(len(draws)),delta_ap=d[:,0],delta_brier=d[:,1],delta_f1=d[:,2])))
            row=dict(contrast=c['name'],left=c['left'],right=c['right'],primary_family=c['primary_family'],
                     block_hours=hours,replicates=len(draws),seed=seed,calendar_blocks=nb,
                     ap_simultaneous_low=observed[i]-critical*sd[i] if c['primary_family'] else np.nan,
                     ap_bootstrap_sd=float(sd[i]),maxT_critical=critical)
            for j,m in enumerate(['ap','brier','f1']):
                row.update({'delta_'+m:estimates[c['left']][m]-estimates[c['right']][m],
                            m+'_low':float(np.quantile(d[:,j],.025)),m+'_high':float(np.quantile(d[:,j],.975))})
            paired_intervals.append(row)
        diagnostics.append(dict(block_hours=hours,blocks=nb,nonempty_blocks=int((counts>0).sum()),seed=seed,n=n,
                                final_block_hours=float((STOP-START)/pd.Timedelta(hours=1)-(nb-1)*hours),
                                unique_draw_vectors=len(np.unique(draws,axis=0)),
                                min_resampled_rows=int((draws@counts).min()),max_resampled_rows=int((draws@counts).max()),
                                min_resampled_positives=int((draws@positives).min()),
                                draw_sha256=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest(),
                                maxT_critical=critical,simultaneous_contrasts=[CONFIG['contrasts'][i]['name'] for i in primary]))
        print('Bootstrap',hours,'hours:',len(draws),'replicates,',nb,'calendar blocks',flush=True)
    save(pd.concat(abs_samples,ignore_index=True),'bootstrap_replicates.parquet')
    save(pd.concat(diff_samples,ignore_index=True),'paired_replicates.parquet')
    save(pd.DataFrame(block_rows),'bootstrap_blocks.csv');write_json(HERE/'bootstrap_diagnostics.json',diagnostics)
    return pd.DataFrame(absolute_intervals),pd.DataFrame(paired_intervals)


def describe(monthly,regional,cells,intervals):
    output={};stability=[];associations=[]
    for v in REDUCED:
        name=v+'_minus_B';g=monthly.loc[monthly.contrast.eq(name)];r=regional.loc[regional.contrast.eq(name)]
        ci=intervals.loc[intervals.contrast.eq(name)]
        output[v]=dict(role='primary frozen fixed list' if v=='C_top5' else 'frozen sensitivity',
                       noninferior_under_registered_rule=bool((ci.ap_low>-.01).all() and (ci.ap_simultaneous_low>-.01).all()),
                       delta_ap=float(ci.delta_ap.iloc[0]),delta_brier=float(ci.delta_brier.iloc[0]),
                       worst_nominal_lower=float(ci.ap_low.min()),worst_simultaneous_lower=float(ci.ap_simultaneous_low.min()),
                       feature_list_unchanged=True,retrospective=True,conditional_inference=True)
        for scope,table in [('month',g),('region',r)]:
            key='origin' if scope=='month' else 'region'
            stability.append(dict(variant=v,scope=scope,count=len(table),ap_improved=int(table.delta_ap.gt(0).sum()),
                                   ap_worse=int(table.delta_ap.lt(0).sum()),brier_improved=int(table.delta_brier.lt(0).sum()),
                                   worst_ap=float(table.delta_ap.min()),worst_stratum=table.loc[table.delta_ap.idxmin(),key],
                                   best_ap=float(table.delta_ap.max()),best_stratum=table.loc[table.delta_ap.idxmax(),key],
                                   mean_delta_ap=float(table.delta_ap.mean())))
        for scope,table in [('month',g),('region',r),('month_region',cells.loc[cells.contrast.eq(name)])]:
            rho=float(spearmanr(table.prevalence,table.delta_ap).statistic)
            associations.append(dict(variant=v,scope=scope,n_strata=len(table),spearman_prevalence_delta_AP=rho,
                                     inference='descriptive only; dependent strata, no significance test'))
    write_json(HERE/'conclusions.json',output);save(pd.DataFrame(stability),'stability.csv')
    save(pd.DataFrame(associations),'prevalence_associations.csv')


def main():
    registered();started=time.monotonic()
    pred=pd.read_parquet(HERE/'predictions.parquet');aligned=align(pred)
    summary,monthly,regional,cells=summaries(aligned)
    save(summary,'summary.csv');save(monthly,'monthly_metrics.csv');save(regional,'regional_metrics.csv');save(cells,'month_region_metrics.csv')
    monthly_c=contrasts_table(monthly,['origin']);regional_c=contrasts_table(regional,['region']);cell_c=contrasts_table(cells,['origin','region'])
    save(monthly_c,'monthly_contrasts.csv');save(regional_c,'regional_contrasts.csv');save(cell_c,'month_region_contrasts.csv')
    absolute,intervals=bootstrap(aligned);save(absolute,'absolute_intervals.csv');save(intervals,'paired_intervals.csv')
    describe(monthly_c,regional_c,cell_c,intervals)
    write_json(HERE/'provenance/analysis.json',dict(utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,protocol_sha256=sha(HERE/'protocol.json'),
        code_sha256=sha(HERE/'analyze_eval.py'),predictions_sha256=sha(HERE/'predictions.parquet')))
    print(json.dumps(json.loads((HERE/'conclusions.json').read_text()),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
