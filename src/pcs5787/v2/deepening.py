"""Ablações, tarefas novas, sementes e estabilidade da calibração/refit."""
import argparse
import fcntl
import json
from copy import deepcopy
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score
from .common import ROOT,P,DATA,OUT,CACHE,REASONS,folds,atomic_json,atomic_frame,note,population,training_mask
from .experiments import Campaign,select,metrics
from .calibration import available_history,fit_calibrator,apply_calibrator,decision_threshold
from .numerical import converged


def quads():
    override=ROOT/'configs/v2/conditional_temporal_selected.json'
    if override.exists():
        result=json.loads(override.read_text())['quad'];atomic_json(ROOT/'configs/v2/quad_selected.json',result);return result
    rows=json.loads((ROOT/'configs/v2/refinement_results.json').read_text())
    rows+=json.loads((ROOT/'configs/v2/synthesis_results.json').read_text())
    result=[]
    selection=[]
    for reason in REASONS:
        for pop in ['anchor_complete','spatial_recent']:
            eligible=[r for r in rows if r['spec']['family']=='coverage_context' and r['spec']['reason']==reason and r['spec']['population']==pop and converged(r)]
            candidates=[]
            for name in sorted({r['spec'].get('candidate','initial') for r in eligible}):
                picks=[]
                for v in ['A','B','C','D']:
                    choices=[r for r in eligible if r['spec'].get('candidate','initial')==name and r['spec']['variant']==v]
                    if choices:picks.append(max(choices,key=lambda z:z['mean_ap']))
                if len(picks)==4:candidates.append((np.mean([p['mean_ap'] for p in picks]),name,picks))
            score,name,picks=max(candidates,key=lambda z:z[0]);result+=picks
            selection.append(dict(reason=reason,population=pop,selected=name,scores={n:float(s) for s,n,_ in candidates},criterion='mean F1-F3 AP, equally weighted A/B/C/D; same algorithm/calendar/window per quad'))
    atomic_json(ROOT/'configs/v2/quad_selection.json',selection)
    atomic_json(ROOT/'configs/v2/quad_selected.json',result)
    return result


def ab_pairs():
    override=ROOT/'configs/v2/conditional_temporal_selected.json'
    if override.exists():
        result=json.loads(override.read_text())['best_AB'];atomic_json(ROOT/'configs/v2/ab_selected.json',result);return result
    rows=json.loads((ROOT/'configs/v2/refinement_results.json').read_text())+json.loads((ROOT/'configs/v2/synthesis_results.json').read_text())
    selected=[]
    for reason in REASONS:
        for pop in ['anchor_complete','spatial_recent']:
            eligible=[r for r in rows if r['spec']['family']=='coverage_context' and r['spec']['reason']==reason and r['spec']['population']==pop and r['spec']['variant'] in ['A','B'] and converged(r)]
            choices=[]
            for name in sorted({r['spec'].get('candidate','initial') for r in eligible}):
                pair=[max([r for r in eligible if r['spec'].get('candidate','initial')==name and r['spec']['variant']==v],key=lambda r:r['mean_ap']) for v in ['A','B']]
                choices.append((np.mean([r['mean_ap'] for r in pair]),pair))
            selected+=max(choices,key=lambda x:x[0])[1]
    atomic_json(ROOT/'configs/v2/ab_selected.json',selected)
    return selected


def ablation_specs():
    best=quads();groups=json.loads((ROOT/'configs/v2/feature_groups.json').read_text());specs=[]
    for row in best:
        s=row['spec']
        if s['population']!='spatial_recent':continue
        for latency in [24,720]:
            specs.append({**s,'family':'availability','latency':latency,'context_suffix':str(latency),'question':'H3/H5 disponibilidade'})
        if s['variant'] in ['B','D']:
            for features,name in [(['basic'],'only_eight'),(['spatial'],'only_spatial'),(['missing'],'missing_only')]:
                specs.append({**s,'family':'ablation_'+name,'features':features,'anomaly':False})
            gust=[c for c in groups['basic']+groups['engineered']+groups['other_weather']+groups['spatial'] if 'gust' in c or 'rajada' in c]
            specs.append({**s,'family':'ablation_without_gust','exclude_columns':gust})
            specs.append({**s,'family':'negative_control_28days','weather_shift_hours':672,'question':'controle temporal, não teste nulo exato'})
        if s['variant'] in ['C','D']:
            specs.append({**s,'family':'ablation_without_cut_history','exclude_columns':[c for c in groups['context'] if any(r in c for r in REASONS)]})
            specs.append({**s,'family':'ablation_without_NE_balance','exclude_columns':[c for c in groups['context'] if '_NE_' in c]})
        specs.append({**s,'family':'reference_floor','min_reference_mwh':10,'question':'robustez do denominador; população restrita'})
    return specs


def tasks_specs():
    out=[]
    for reason in ['onset','end','widespread','magnitude']:
        for strategy in ['window12','half_life180']:
            for leaves,min_leaf,l2 in [(7,100,5),(15,100,10),(31,200,20)]:
                for variant in ['A','B','C','D']:
                    for objective in (['regression','regression_l1'] if reason=='magnitude' else ['binary']):
                        out.append(dict(family='task_'+reason,question={'onset':'H5','end':'H5','widespread':'H6','magnitude':'H5'}[reason],reason=reason,variant=variant,population='spatial_recent',features=['basic','engineered','other_weather','spatial'],calendar='strong',algorithm='lightgbm',params=dict(objective=objective,leaves=leaves,min_leaf=min_leaf,l2=l2,iterations=250,learning_rate=.05),strategy=strategy,seed=5787))
    return out


def task_search(c):
    specs=tasks_specs();atomic_json(ROOT/'configs/v2/tasks_grid.json',specs);rows=[]
    for s in specs:
        runs=[c.run(s,f)[0] for f in folds()]
        key='mae' if s['reason']=='magnitude' else 'ap'
        scores=[r['metrics'][key] for r in runs]
        rows.append(dict(spec=s,metric=key,mean_score=float(np.nanmean(scores)),fold_scores=scores,run_ids=[r['run_id'] for r in runs]))
    atomic_json(ROOT/'configs/v2/tasks_results.json',rows)
    selected=[]
    for reason in ['onset','end','widespread','magnitude']:
        minimize=reason=='magnitude';choices=[]
        for strategy in ['window12','half_life180']:
            picks=[]
            for v in ['A','B','C','D']:
                candidates=[r for r in rows if r['spec']['reason']==reason and r['spec']['variant']==v and r['spec']['strategy']==strategy]
                picks.append((min if minimize else max)(candidates,key=lambda z:z['mean_score']))
            choices.append((np.mean([p['mean_score'] for p in picks]),picks))
        selected += (min if minimize else max)(choices,key=lambda x:x[0])[1]
    atomic_json(ROOT/'configs/v2/tasks_selected.json',selected)


def calibration_search(c):
    rows=[];choices=[];oof=[]
    for panel,result in [('quad',r) for r in quads()]+[('best_AB',r) for r in ab_pairs()]:
        s=result['spec'];runs=[c.run(s,f)[1] for f in folds()]
        all_oof=pd.concat(runs,ignore_index=True)
        for f,pred in zip(folds(),runs):
            d=pred.copy();d['fold']=f['name'];d['reason']=s['reason'];d['variant']=s['variant'];d['population']=s['population'];d['algorithm']=s['algorithm'];d['seed']=s['seed'];d['panel']=panel;oof.append(d)
        for i in [1,2]:
            f=folds()[i];history=available_history(pd.concat(runs[:i]),f['origin'])
            # Sem refit: mesmo estimador que gerou as probabilidades usadas para calibrar.
            holdfold={**f,'name':'hold_'+f['name'],'origin':folds()[i-1]['origin'],'train_stop':folds()[i-1]['train_stop']}
            hold=c.run({**s,'family':'calibration_hold_model'},holdfold)[1]
            for method in ['none','intercept','monotone_platt']:
                cal=fit_calibrator(history.y,history.p,method)
                threshold=decision_threshold(history.y,apply_calibrator(history.p,cal))
                for system,d in [('refit',runs[i]),('hold_model',hold)]:
                    p=apply_calibrator(d.p,cal)
                    rows.append({'panel':panel,'reason':s['reason'],'population':s['population'],'variant':s['variant'],'fold':f['name'],'system':system,'method':method,'slope':cal['slope'],'intercept':cal['intercept'],'n_calibration':len(history),'calibration_end':str(history.time_local.max()),**metrics(d.y,p,threshold)})
        d=pd.DataFrame(rows);d=d[(d.panel==panel)&(d.reason==s['reason'])&(d.population==s['population'])&(d.variant==s['variant'])&d.system.eq('refit')]
        score=d.groupby('method').brier.mean();method=score.idxmin()
        choices.append({'panel':panel,'reason':s['reason'],'population':s['population'],'variant':s['variant'],'method':method,'validation_brier':score.to_dict()})
    atomic_frame(pd.DataFrame(rows),OUT/'tables/calibration_validation.csv')
    atomic_frame(pd.concat(oof,ignore_index=True),DATA/'selected_oof.parquet')
    atomic_json(ROOT/'configs/v2/calibration_selected.json',choices)


def stability(c):
    specs=[]
    for r in quads()+ab_pairs():
        s=r['spec']
        stochastic=s['algorithm']=='lightgbm' or (s['algorithm']=='histgb' and s['strategy'] in ['expanding','half_life180'])
        if s['population']!='spatial_recent' or not stochastic:continue
        for seed in P['final_stochastic_seeds'][1:]:specs.append({**s,'family':'seed_stability','seed':seed})
    specs=list({json.dumps(s,sort_keys=True):s for s in specs}.values())
    select(specs,c,'stability_results')
    task_rows=[]
    for result in json.loads((ROOT/'configs/v2/tasks_selected.json').read_text()):
        for seed in P['final_stochastic_seeds'][1:]:
            spec={**result['spec'],'family':'task_seed_stability','seed':seed}
            runs=[c.run(spec,f)[0] for f in folds()]
            metric='mae' if spec['reason']=='magnitude' else 'ap'
            values=[r['metrics'][metric] for r in runs]
            task_rows.append(dict(spec=spec,metric=metric,mean_score=float(np.mean(values)),fold_scores=values,run_ids=[r['run_id'] for r in runs]))
    atomic_json(ROOT/'configs/v2/task_stability_results.json',task_rows)


def learning_curves(c):
    rows=[]
    for result in quads():
        s=result['spec']
        if s['population']!='spatial_recent':continue
        for months in [3,6,12,24]:
            spec={**s,'family':'learning_window','strategy':f'window{months}','question':'learning/recency diagnostic, fixed F3 and parameters; not selection'}
            meta,_=c.run(spec,folds()[-1],save_model=True)
            eligible,target=population(c.base,s['reason'],s['population'])
            tr=c.base[eligible&training_mask(c.base,folds()[-1]['origin'],spec['strategy'])]
            rng=np.random.default_rng(P['seed']);sample=tr.iloc[rng.choice(len(tr),min(12000,len(tr)),replace=False)]
            model=joblib.load(ROOT/f'models/v2/{meta["run_id"]}.joblib')
            training_ap=average_precision_score(sample[target],model.predict_proba(sample)[:,1])
            rows.append(dict(reason=s['reason'],variant=s['variant'],algorithm=s['algorithm'],months=months,n_train=meta['n_train'],training_sample_ap=training_ap,training_sample_n=len(sample),training_sample_regions=sample.region.nunique(),training_sampling='uniform without replacement, fixed seed; training rows only',validation_ap=meta['metrics']['ap'],run_id=meta['run_id']))
    atomic_frame(pd.DataFrame(rows),OUT/'tables/learning_curves.csv')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['ablation','tasks','calibration','stability','learning'],required=True);args=parser.parse_args()
    with (ROOT/'.scratch/v2/campaign.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);c=Campaign()
        if args.stage=='ablation':
            specs=ablation_specs();atomic_json(ROOT/'configs/v2/ablation_grid.json',specs);select(specs,c,'ablation_results')
        elif args.stage=='tasks':task_search(c)
        elif args.stage=='calibration':calibration_search(c)
        elif args.stage=='learning':learning_curves(c)
        else:stability(c)
        note('R3 '+args.stage,'Etapa concluída nas validações F1–F3. Tabelas/configurações persistidas; o bloco externo ainda não foi usado para escolher modelos.')


if __name__=='__main__':main()
