"""Congelamento e avaliação externa; nunca seleciona pelo desempenho externo."""
import json
import time
import fcntl
from copy import deepcopy
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from ..common import ROOT,sha256
from .common import P,DATA,OUT,CACHE,REASONS,atomic_json,atomic_frame,stamp,cutoff,population,training_mask,digest,note
from .experiments import Campaign,metrics
from .deepening import quads,ab_pairs
from .calibration import available_history,fit_calibrator,apply_calibrator,decision_threshold
from .baselines import CalendarFrequency


def external_fold(origin='2026-04-01',stop='2026-10-01',name='external'):
    return dict(name=name,origin=origin,start=stamp(origin),stop=stamp(stop),train_stop=cutoff(origin))


def freeze_finalists():
    path=ROOT/'configs/v2/finalists_frozen.json'
    if path.exists():
        old=json.loads(path.read_text())
        payload={k:v for k,v in old.items() if k not in ['frozen_at_utc','external_scores_inspected_in_v2','historically_blind','fingerprint']}
        assert digest(payload)==old['fingerprint'],'Manifesto congelado foi alterado'
        assert old['protocol_sha256']==sha256(ROOT/'configs/v2/protocol.json') and old['dataset_sha256']==sha256(DATA/'analytical.parquet')
        assert old['feature_groups_sha256']==sha256(ROOT/'configs/v2/feature_groups.json')
        for name,value in old['engine_sha256'].items():assert sha256(ROOT/'src/pcs5787/v2'/name)==value
        return old # Applying frozen choices must not rerun or reinterpret selection.
    selected=quads()
    tasks=json.loads((ROOT/'configs/v2/tasks_selected.json').read_text())
    cal=json.loads((ROOT/'configs/v2/calibration_selected.json').read_text())
    availability=[r for r in json.loads((ROOT/'configs/v2/ablation_results.json').read_text()) if r['spec']['family']=='availability']
    ensembles=json.loads((ROOT/'configs/v2/ensemble_selected.json').read_text())
    payload={'quad':selected,'best_AB':ab_pairs(),'availability':availability,'tasks':tasks,'calibration':cal,'protocol_sha256':sha256(ROOT/'configs/v2/protocol.json'),'dataset_sha256':sha256(DATA/'analytical.parquet'),
             'ensembles':ensembles,
             'feature_groups_sha256':sha256(ROOT/'configs/v2/feature_groups.json'),'engine_sha256':{name:sha256(ROOT/'src/pcs5787/v2'/name) for name in ['common.py','learners.py','experiments.py']},
             'contrast_plan':['P1_access_only minus P0_logistic: same V1 architecture and population','P2_access_only minus P1_access_only: monthly updates with same architecture','P1/P2 B-A and D-C within identical population','H2 gain differences on common reasons','P0_intercept vs P0: probability correction without reranking','P1_holdcal vs P1: calibration compatibility after refit','ancillary tasks: onset, termination, spatial dissemination, fraction magnitude'],
             'robustness':['seeds 5787/1909/2718 for stochastic P1 spatial population','temporal blocks 168/672h','physical labels tau1/5 evaluated without retuning probabilities','reference >=10 MWh subset without changing probabilities','region/month/covariate support strata','development ablations preserved; no final hyperparameter reselection'],
             'primary_ranking':'AP from raw probabilities; calibrated AP also reported; P2 includes pooled and monthly AP',
             'hypothesis_panels':'H1/H2: best A/B family selected by mean A/B validation AP (P1_AB/P2_AB), alongside shared A/B/C/D family selected by mean quad AP (P1/P2); do not claim meteorological gain solely from a weaker calendar baseline',
             'ancillary_transfer':'H6: compare widespread-specific model with ENE model score, evaluating both against widespread labels on shared hours; AP is comparable only after relabeling the scoring target consistently',
             'solver_rule':'Selected logistic fits converged during development; final V2 logistic fits allow 3000 iterations with unchanged tolerance, objective and regularization. This is a convergence ceiling, not model reselection. V1 clones keep original solver settings.',
             'article_and_slides':'remain explicitly V1; V2 prioritizes investigation and aligned weekly reports'}
    fp=digest(payload)
    frozen={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'external_scores_inspected_in_v2':False,'historically_blind':False,'fingerprint':fp,**payload}
    atomic_json(path,frozen)
    return frozen


def decorate(d,s,case,cal=None,threshold=.5,fold='external',model_id=''):
    out=d.copy();out=out.rename(columns={'p':'p_raw'})
    if cal is None:cal={'method':'none','slope':1.,'intercept':0.}
    out['p']=apply_calibrator(out.p_raw,cal)
    out['threshold']=threshold;out['reason']=s['reason'];out['variant']=s.get('variant','A');out['population']=s.get('population','anchor_complete');out['protocol']=case;out['fold']=fold;out['algorithm']=s.get('algorithm','logistic');out['seed']=s.get('seed',5787);out['model_id']=model_id;out['calibration']=cal['method']
    return out


def model_origins(protocol):
    if protocol=='P1':return [('2026-04-01','2026-10-01')]
    starts=P['sequential_origins'];return list(zip(starts,starts[1:]+['2026-10-01']))


def main():
    with (ROOT/'.scratch/v2/campaign.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=freeze_finalists();c=Campaign();b=c.base
        oof=pd.read_parquet(DATA/'selected_oof.parquet')
        availability_oof=pd.read_parquet(DATA/'availability_oof.parquet')
        tasks_oof=pd.read_parquet(DATA/'tasks_oof.parquet')
        ensemble_oof=pd.read_parquet(DATA/'ensemble_oof.parquet')
        outputs=[];calibration_records=[];split_records=[];extra_ledger=[]
        def split_record(s,origin,role):
            eligible,target=population(b,s['reason'],s.get('population','anchor_complete'))
            tr=b[eligible&training_mask(b,origin,s['strategy'],s.get('latency'))]
            for yr,g in tr.groupby('year'):
                weight=np.exp2(-(cutoff(origin,s.get('latency'))-g.time_local).dt.total_seconds().to_numpy()/(86400*180)) if s['strategy']=='half_life180' else np.ones(len(g))
                split_records.append(dict(protocol=role,origin=origin,reason=s['reason'],population=s.get('population','anchor_complete'),strategy=s['strategy'],year=yr,n=len(g),positives=int(g[target].sum()) if s['reason']!='magnitude' else np.nan,weight_sum=weight.sum(),weight_square_sum=np.sum(weight**2),first=g.time_local.min(),last=g.time_local.max(),keys_sha256=digest(pd.util.hash_pandas_object(g[['region','time_local']],index=False).tolist())))
        # Quatro conjuntos aninhados; P1/P2 com o mesmo orçamento e população de cada par.
        for panel,chosen in [('quad',r) for r in frozen['quad']]+[('best_AB',r) for r in frozen['best_AB']]:
            s=deepcopy(chosen['spec'])
            if s['algorithm'].startswith('logistic'):s['params']['max_iter']=3000
            method=next(x['method'] for x in frozen['calibration'] if x['panel']==panel and all(x[k]==s[k] for k in ['reason','population','variant']))
            past=oof[(oof.panel==panel)&(oof.reason==s['reason'])&(oof.population==s['population'])&(oof.variant==s['variant'])&oof.fold.eq('F3')][['time_local','region','y','p']].copy()
            for protocol in ['P1','P2']:
                case=protocol+('_AB' if panel=='best_AB' else '')
                pool=past.copy()
                for origin,end in model_origins(protocol):
                    f=external_fold(origin,end,protocol+'_'+origin[:7]);fit_spec={**s,'family':'final_'+protocol}
                    meta,d=c.run(fit_spec,f,save_model=True)
                    assert not any('failed to converge' in w.lower() for w in meta.get('warnings',[])),meta['run_id']
                    history=available_history(pool,origin)
                    cal=fit_calibrator(history.y,history.p,method)
                    threshold=decision_threshold(history.y,apply_calibrator(history.p,cal))
                    outputs.append(decorate(d,s,case,cal,threshold,f['name'],meta['run_id']))
                    calibration_records.append(dict(protocol=case,origin=origin,reason=s['reason'],population=s['population'],variant=s['variant'],history_first=history.time_local.min(),history_last=history.time_local.max(),threshold=threshold,**cal))
                    if s['variant']=='A':split_record(s,origin,case)
                    pool=pd.concat([pool,d[['time_local','region','y','p']]],ignore_index=True)
            # Replicações estocásticas, sem selecionar a seed pelo externo.
            stochastic=s['algorithm']=='lightgbm' or (s['algorithm']=='histgb' and s['strategy'] in ['expanding','half_life180'])
            ensemble_seeds=any(z['promote'] and z['reason']==s['reason'] and z['population']==s['population'] for z in frozen['ensembles'])
            if (s['population']=='spatial_recent' or ensemble_seeds) and stochastic:
                for seed in P['final_stochastic_seeds'][1:]:
                    spec={**s,'family':'final_P1_seed','seed':seed};meta,d=c.run(spec,external_fold(),save_model=True)
                    history=available_history(past,'2026-04-01');cal=fit_calibrator(history.y,history.p,method)
                    threshold=decision_threshold(history.y,apply_calibrator(history.p,cal))
                    outputs.append(decorate(d,spec,'P1'+('_AB' if panel=='best_AB' else ''),cal,threshold,model_id=meta['run_id']))
            # Controle sem refit: modelo de janeiro continua após sua calibração F3.
            hold={**external_fold(),'name':'external_holdcal','origin':'2026-01-01','train_stop':cutoff('2026-01-01')}
            meta,d=c.run({**s,'family':'final_holdcal'},hold,save_model=True)
            history=available_history(past,'2026-04-01');cal=fit_calibrator(history.y,history.p,method)
            outputs.append(decorate(d,s,'P1'+('_AB' if panel=='best_AB' else '')+'_holdcal',cal,decision_threshold(history.y,apply_calibrator(history.p,cal)),model_id=meta['run_id']))
        # Predeclared publication scenarios, evaluated with the same selected capacity.
        for result in frozen['availability']:
            s=deepcopy(result['spec']);latency=s['latency']
            if s['algorithm'].startswith('logistic'):s['params']['max_iter']=3000
            f=external_fold();f.update(name=f'external_latency{latency}',train_stop=cutoff('2026-04-01',latency))
            meta,d=c.run({**s,'family':'final_availability'},f,save_model=True)
            assert not any('failed to converge' in w.lower() for w in meta.get('warnings',[])),meta['run_id']
            past=availability_oof[availability_oof.reason.eq(s['reason'])&availability_oof.variant.eq(s['variant'])&availability_oof.latency.eq(latency)&availability_oof.fold.eq('F3')]
            history=available_history(past,'2026-04-01',latency=latency)
            method=next(x['method'] for x in frozen['calibration'] if x['panel']=='quad' and all(x[k]==s[k] for k in ['reason','population','variant']))
            cal=fit_calibrator(history.y,history.p,method)
            outputs.append(decorate(d,s,f'P1_latency{latency}',cal,decision_threshold(history.y,apply_calibrator(history.p,cal)),model_id=meta['run_id']))
            calibration_records.append(dict(protocol=f'P1_latency{latency}',origin='2026-04-01',reason=s['reason'],population=s['population'],variant=s['variant'],history_first=history.time_local.min(),history_last=history.time_local.max(),**cal))
            if s['variant']=='A':split_record(s,'2026-04-01',f'P1_latency{latency}')
        # Convex mixtures only when they passed the development gate; weights are shared.
        component_predictions=pd.concat(outputs,ignore_index=True) if any(z['promote'] for z in frozen['ensembles']) else None
        for choice in frozen['ensembles']:
            if not choice['promote']:continue
            r,pop,alpha=choice['reason'],choice['population'],choice['alpha_AB']
            for protocol in ['P1','P2']:
                for v in ['A','B','C','D']:
                    past=ensemble_oof[ensemble_oof.reason.eq(r)&ensemble_oof.population.eq(pop)&ensemble_oof.variant.eq(v)&ensemble_oof.fold.eq('F3')][['time_local','region','y','p']]
                    seeds=sorted(component_predictions.loc[component_predictions.protocol.isin(['P1','P1_AB'])&component_predictions.reason.eq(r)&component_predictions.population.eq(pop),'seed'].unique().tolist()) if protocol=='P1' else [5787]
                    for seed in seeds:
                        pool=past.copy()
                        for origin,end in model_origins(protocol):
                            start_mixture=time.monotonic()
                            parts=[]
                            for case,variant in [(protocol+'_AB','A' if v in ['A','C'] else 'B'),(protocol,v)]:
                                z=component_predictions[component_predictions.protocol.eq(case)&component_predictions.reason.eq(r)&component_predictions.population.eq(pop)&component_predictions.variant.eq(variant)&component_predictions.time_local.ge(stamp(origin))&component_predictions.time_local.lt(stamp(end))]
                                target_seed=seed if z.seed.eq(seed).any() else 5787
                                if target_seed!=seed:assert not z.algorithm.eq('lightgbm').any(),'Missing stochastic component seed'
                                parts.append(z[z.seed.eq(target_seed)])
                            left,right=parts
                            d=left[['time_local','region','y','p_raw']].merge(right[['time_local','region','y','p_raw']],on=['time_local','region'],suffixes=('_l','_r'),validate='one_to_one')
                            np.testing.assert_array_equal(d.y_l,d.y_r)
                            mixed=d[['time_local','region']].copy();mixed['y']=d.y_l;mixed['p']=alpha*d.p_raw_l+(1-alpha)*d.p_raw_r
                            modelinfo=dict(alpha_AB=alpha,left_models=sorted(left.model_id.unique().tolist()),right_models=sorted(right.model_id.unique().tolist()),rule='mixture of raw probabilities; no target or contemporaneous energy input')
                            ident='mixture_'+digest(modelinfo)[:20];atomic_json(ROOT/f'models/v2/{ident}.json',modelinfo)
                            history=available_history(pool,origin);cal=fit_calibrator(history.y,history.p,choice['calibration'][v])
                            threshold=decision_threshold(history.y,apply_calibrator(history.p,cal))
                            s=dict(reason=r,population=pop,variant=v,algorithm='convex_mixture',seed=seed)
                            outputs.append(decorate(mixed,s,protocol+'_ensemble',cal,threshold,origin[:7],ident))
                            extra_ledger.append(dict(run_id=ident,family='final_ensemble',operation='combination_and_calibration',question='complementarity of strong calendar and contextual family',reason=r,population=pop,dataset_sha256=c.dataset,variant=v,features='raw component predictions',fold=protocol+'_'+origin[:7],strategy='fixed convex weights; past calibration',algorithm='convex_mixture',params=json.dumps(modelinfo),seed=seed,n_calibration=len(history),seconds=time.monotonic()-start_mixture,status='done',**metrics(mixed.y,apply_calibrator(mixed.p,cal),threshold)))
                            if seed==5787:calibration_records.append(dict(protocol=protocol+'_ensemble',origin=origin,reason=r,population=pop,variant=v,history_first=history.time_local.min(),history_last=history.time_local.max(),threshold=threshold,**cal))
                            pool=pd.concat([pool,mixed],ignore_index=True)
        # Referência V1 preservada, incluindo famílias secundárias e constante.
        v1=pd.read_parquet(ROOT/'data/processed/v1_reference/predictions.parquet')
        ext=v1[v1.time_local.ge(stamp(P['final_origin']))&v1.time_local.lt(stamp(P['final_end_exclusive']))]
        for (r,alg,v),d in ext.groupby(['reason','algorithm','variant']):
            s=dict(reason=r,algorithm=alg,variant=v,population='v1_common',seed=5787)
            outputs.append(decorate(d[['time_local','region','y','p']],s,'P0_'+alg,threshold=d.threshold.iloc[0],model_id='V1_17414c1'))
            if alg=='logistic':
                history=available_history(v1[(v1.reason==r)&(v1.algorithm==alg)&(v1.variant==v)&v1.time_local.ge(stamp('2026-01-01'))],'2026-04-01')
                cal=fit_calibrator(history.y,history.p,'intercept')
                outputs.append(decorate(d[['time_local','region','y','p']],s,'P0_intercept',cal,decision_threshold(history.y,apply_calibrator(history.p,cal)),model_id='V1_17414c1'))
        # Efeito puro de incorporar rótulos recentes: clonar a arquitetura/hiperparâmetros V1.
        for r in REASONS:
            for v in ['A','B']:
                original=ROOT/f'models/v1_reference/{r}_logistic_{v}.joblib'
                info=json.loads((ROOT/f'models/v1_reference/{r}_logistic_{v}.json').read_text())
                s=dict(reason=r,variant=v,algorithm='V1_logistic_clone',population='v1_common',strategy='expanding',seed=5787)
                eligible,target=population(b,r,'v1_common')
                for protocol in ['P1','P2']:
                    for origin,end in model_origins(protocol):
                        start=time.monotonic();tr=b[eligible&training_mask(b,origin,'expanding')];va=b[eligible&b.time_local.ge(stamp(origin))&b.time_local.lt(stamp(end))]
                        ident=digest([s,protocol,origin,end,c.dataset,sha256(original),sha256(__file__)])[:20]
                        mp=ROOT/f'models/v2/{ident}.joblib';pp=CACHE/f'{ident}.parquet';jp=CACHE/f'{ident}.json'
                        if pp.exists() and mp.exists() and jp.exists():
                            record=json.loads(jp.read_text());assert record['pred_sha256']==sha256(pp) and record['model_sha256']==sha256(mp);d=pd.read_parquet(pp)
                        else:
                            model=clone(joblib.load(original))
                            with threadpool_limits(limits=4):model.fit(tr,tr[target].astype(int));p=model.predict_proba(va)[:,1]
                            d=va[['time_local','region']].copy();d['y']=va[target].to_numpy();d['p']=p
                            atomic_frame(d,pp);tmp=mp.with_suffix('.tmp');joblib.dump(model,tmp,compress=3);tmp.replace(mp)
                            record={'pred_sha256':sha256(pp),'model_sha256':sha256(mp),'seconds':time.monotonic()-start};atomic_json(jp,record)
                        outputs.append(decorate(d,s,protocol+'_access_only',threshold=info['threshold'],fold=origin[:7],model_id=ident))
                        extra_ledger.append(dict(run_id=ident,family='access_only',question='H3: same V1 architecture, recent labels only',reason=r,population='v1_common',dataset_sha256=c.dataset,variant=v,features=json.dumps(['calendar']+(['V1_eight'] if v=='B' else [])),fold=protocol+'_'+origin[:7],strategy='expanding',algorithm='V1_logistic_clone',params=json.dumps({'C':info['parameter'],'threshold_original':info['threshold']}),seed=5787,n_train=len(tr),train_start=tr.time_local.min(),train_end=tr.time_local.max(),seconds=record['seconds'],status='done',**metrics(d.y,d.p,info['threshold'])))
                        if v=='A':split_record(s,origin,protocol+'_access_only')
        # Referências fortes na mesma população; são previsões anteriores à observação alvo.
        frequency=json.loads((ROOT/'configs/v2/frequency_selected.json').read_text())
        for choice in frequency:
            r,pop=choice['reason'],choice['population'];eligible,target=population(b,r,pop)
            for protocol in ['P1','P2']:
                for origin,end in model_origins(protocol):
                    tr=b[eligible&training_mask(b,origin,choice['strategy'])];va=b[eligible&b.time_local.ge(stamp(origin))&b.time_local.lt(stamp(end))]
                    freq=CalendarFrequency(choice['alpha']).fit(tr,tr[target]);prior=tr.loc[tr.time_local.ge(cutoff(origin)-pd.Timedelta(days=90)),target].mean()
                    if not np.isfinite(prior):prior=tr[target].mean()
                    predictions={'frequency':freq.predict(va),'recent_prior':np.full(len(va),prior),'persistence170':va[f'op_{r}_past'].fillna(prior).clip(.01,.99).to_numpy()}
                    for name,p in predictions.items():
                        s=dict(reason=r,population=pop,algorithm=name,variant='T',seed=5787)
                        d=va[['time_local','region']].copy();d['y']=va[target].to_numpy();d['p']=p
                        outputs.append(decorate(d,s,protocol+'_'+name,fold=origin[:7]))
        # Novas tarefas têm definições próprias; não comparar sua AP diretamente à ocorrência.
        for result in frozen['tasks']:
            s=result['spec'];meta,d=c.run({**s,'family':'final_task'},external_fold(),save_model=True)
            past=tasks_oof[tasks_oof.reason.eq(s['reason'])&tasks_oof.variant.eq(s['variant'])&tasks_oof.fold.eq('F3')]
            threshold=decision_threshold(past.y,past.p) if s['reason']!='magnitude' else np.nan
            outputs.append(decorate(d,s,'P1_task',threshold=threshold,model_id=meta['run_id']))
            for seed in P['final_stochastic_seeds'][1:]:
                replicate={**s,'family':'final_task_seed','seed':seed}
                seedmeta,seedd=c.run(replicate,external_fold(),save_model=True)
                outputs.append(decorate(seedd,replicate,'P1_task',threshold=threshold,model_id=seedmeta['run_id']))
            if s['variant']=='A':
                eligible,target=population(b,s['reason'],s['population'])
                tr=b[eligible&training_mask(b,'2026-04-01',s['strategy'])]
                for name,value in [('zero',0.),('training_mean',tr[target].mean()),('training_median',tr[target].median())]:
                    reference=d.copy();reference['p']=value
                    outputs.append(decorate(reference,{**s,'algorithm':name,'variant':'T'},'P1_task_'+name,threshold=.5,model_id='past_only_baseline'))
        final=pd.concat(outputs,ignore_index=True)
        assert not final.duplicated(['protocol','population','reason','variant','seed','time_local','region']).any()
        final['dataset_version']=c.dataset[:16]
        atomic_frame(final,DATA/'final_predictions.parquet')
        atomic_frame(pd.DataFrame(calibration_records),OUT/'tables/final_calibration_history.csv')
        atomic_frame(pd.DataFrame(split_records),OUT/'tables/final_training_composition.csv')
        ledger=pd.read_csv(c.ledger_path);extra=pd.DataFrame(extra_ledger);ledger=ledger[~ledger.run_id.isin(extra.run_id)]
        atomic_frame(pd.concat([ledger,extra],ignore_index=True),c.ledger_path)
        atomic_json(OUT/'tables/external_execution.json',{'completed_at_utc':datetime.now(timezone.utc).isoformat(),'frozen_finalists_sha256':sha256(ROOT/'configs/v2/finalists_frozen.json'),'controller_sha256':sha256(__file__),'prediction_sha256':sha256(DATA/'final_predictions.parquet'),'rows':len(final),'historically_blind':False})
        note('R4 previsões externas','P0, P1, P2, clones V1 com dados recentes, recalibração/sem refit, referências e novas tarefas executados. Predições individuais persistidas; nenhuma escolha de configuração realizada pelo externo. Próximo: métricas pareadas, incerteza e contestação.')
        print('Previsões externas:',len(final),flush=True)


if __name__=='__main__':main()
