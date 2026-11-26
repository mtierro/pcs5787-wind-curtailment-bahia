"""Paired development metrics and bootstrap; fixed pre-registered choice rule."""
from datetime import datetime, timezone
import json
import resource
import time
import numpy as np
import pandas as pd
from audit import HERE, ROOT, sha, write_json, check_protocol
from selection_core import (CONFIG, FEATURES, VARIANTS, REDUCED, FIRST, DEV_STOP,
                            WeightedAP, strictly_align, score)
from campaign import save_frame


def summaries(pred, ledger):
    monthly=[]; summary=[]; deltas=[]
    for variant in VARIANTS:
        frame=pred.loc[pred.variant.eq(variant)]
        model_rows=ledger.loc[ledger.variant.eq(variant)]
        rows=[]
        for origin, sub in frame.groupby("origin",sort=True):
            metrics=score(sub.y,sub.p,sub.threshold)
            row=dict(variant=variant,origin=origin,threshold=float(sub.threshold.iloc[0]),**metrics)
            rows.append(row);monthly.append(row)
        by_month=pd.DataFrame(rows)
        pooled=score(frame.y,frame.p,frame.threshold)
        summary.append(dict(variant=variant,**pooled,
            mean_monthly_ap=float(by_month.ap.mean()),sd_monthly_ap=float(by_month.ap.std()),
            min_monthly_ap=float(by_month.ap.min()),max_monthly_ap=float(by_month.ap.max()),
            mean_monthly_brier=float(by_month.brier.mean()),
            n_features_min=int(model_rows.n_features.min()),n_features_max=int(model_rows.n_features.max()),
            n_features_mean=float(model_rows.n_features.mean()),effective_entries_min=int(model_rows.effective_entries.min()),
            effective_entries_max=int(model_rows.effective_entries.max()),
            fit_seconds_total=float(model_rows.fit_seconds.sum()),fit_seconds_mean=float(model_rows.fit_seconds.mean()),
            predict_ms_per_1000_mean=float(model_rows.predict_ms_per_1000.mean())))
    summary=pd.DataFrame(summary); monthly=pd.DataFrame(monthly)
    baseline=monthly.loc[monthly.variant.eq("B")].set_index("origin")
    for row in monthly.itertuples():
        if row.variant=="B":continue
        reference=baseline.loc[row.origin]
        deltas.append(dict(variant=row.variant,origin=row.origin,n=row.n,
                           delta_ap=row.ap-reference.ap,delta_brier=row.brier-reference.brier,
                           delta_f1=row.f1-reference.f1,delta_alert_fraction=row.alert_fraction-reference.alert_fraction))
    return summary,monthly,pd.DataFrame(deltas)


def historical_checks(aligned):
    old={"2025-01":("591ad7d57961cce0b0df","00fc1de073369b535078"),
         "2025-07":("ef28660940fc1722d193","a8244fbf71823588a643"),
         "2026-01":("264af0a1c9d2e425b94d","188914291adf201d7581")}
    rows=[]
    for origin, ids in old.items():
        for variant,run_id in zip(["A","B"],ids):
            current=aligned[variant].loc[aligned[variant].origin.eq(origin)].sort_values(["time_local","region"]).reset_index(drop=True)
            # Development cache only: no final binary or final prediction is opened.
            historical=pd.read_parquet(ROOT/f".scratch/v2/cache/{run_id}.parquet")
            assert historical.time_local.max()<DEV_STOP
            historical=historical.loc[historical.time_local.ge(current.time_local.min()) & historical.time_local.le(current.time_local.max())].sort_values(["time_local","region"]).reset_index(drop=True)
            assert len(current)==len(historical)
            for col in ["time_local","region","y"]:
                assert np.array_equal(current[col].to_numpy(),historical[col].to_numpy())
            residual=float(np.max(np.abs(current.p.to_numpy()-historical.p.to_numpy())))
            assert residual<=CONFIG["validation_tolerances"]["probability"]
            rows.append(dict(origin=origin,variant=variant,historical_run_id=run_id,n=len(current),
                             probability_max_residual=residual,
                             interpretation="new monthly run versus first month of corresponding historical F1/F2/F3; not equality of validation protocols"))
    return pd.DataFrame(rows)


def bootstrap(aligned):
    ref=aligned["B"]
    y=ref.y.to_numpy(int);n=len(ref)
    names=[v for v in VARIANTS if v!="B"]
    scorers={v:WeightedAP(y,aligned[v].p.to_numpy()) for v in VARIANTS}
    ones=np.ones(n,dtype=int)
    original_ap={v:scorers[v](ones) for v in VARIANTS}
    original_brier={v:float(np.mean((aligned[v].p.to_numpy()-y)**2)) for v in VARIANTS}
    all_samples=[];intervals=[];diagnostics=[]
    for hours in CONFIG["bootstrap"]["block_hours"]:
        blocks=((ref.time_local-FIRST).dt.total_seconds().to_numpy()/3600//hours).astype(int)
        nblocks=int(np.ceil((DEV_STOP-FIRST)/pd.Timedelta(hours=hours)))
        assert blocks.min()==0 and blocks.max()<nblocks
        counts=np.bincount(blocks,minlength=nblocks)
        seed=5787+hours
        draws=np.random.default_rng(seed).multinomial(nblocks,np.full(nblocks,1/nblocks),size=CONFIG["bootstrap"]["replicates"])
        assert np.array_equal(draws,np.random.default_rng(seed).multinomial(nblocks,np.full(nblocks,1/nblocks),size=CONFIG["bootstrap"]["replicates"]))
        losses={v:np.bincount(blocks,weights=(aligned[v].p.to_numpy()-y)**2,minlength=nblocks) for v in VARIANTS}
        ap_samples=np.zeros((len(draws),len(names)));br_samples=np.zeros_like(ap_samples)
        for rep,draw in enumerate(draws):
            w=draw[blocks];ap_b=scorers["B"](w);den=float(draw@counts);br_b=float(draw@losses["B"])/den
            for j,variant in enumerate(names):
                ap_samples[rep,j]=scorers[variant](w)-ap_b
                br_samples[rep,j]=float(draw@losses[variant])/den-br_b
            if (rep+1)%500==0: print(f"Bootstrap {hours} h: {rep+1}/{len(draws)}",flush=True)
        assert np.isfinite(ap_samples).all() and np.isfinite(br_samples).all()
        eligible_idx=[names.index(v) for v in REDUCED]
        observed=np.array([original_ap[v]-original_ap["B"] for v in names])
        std=ap_samples.std(axis=0,ddof=1)
        centered=ap_samples[:,eligible_idx]-observed[eligible_idx]
        standardized=np.divide(centered,std[eligible_idx],out=np.zeros_like(centered),where=std[eligible_idx]>0)
        critical=float(np.quantile(standardized.max(axis=1),.975))
        for j,variant in enumerate(names):
            lo,hi=np.quantile(ap_samples[:,j],[.025,.975]);blo,bhi=np.quantile(br_samples[:,j],[.025,.975])
            simultaneous=observed[j]-critical*std[j] if variant in REDUCED else np.nan
            intervals.append(dict(variant=variant,block_hours=hours,delta_ap=observed[j],
                                  ap_low=float(lo),ap_high=float(hi),ap_bootstrap_sd=float(std[j]),
                                  ap_simultaneous_low=float(simultaneous),maxT_critical=critical,
                                  delta_brier=original_brier[variant]-original_brier["B"],
                                  brier_low=float(blo),brier_high=float(bhi),
                                  nominal_noninferior=bool(lo>-.01),
                                  simultaneous_noninferior=bool(simultaneous>-.01) if variant in REDUCED else False,
                                  calendar_blocks=nblocks,replicates=len(draws),seed=seed))
            all_samples.append(pd.DataFrame(dict(variant=variant,block_hours=hours,rep=np.arange(len(draws)),
                                                 delta_ap=ap_samples[:,j],delta_brier=br_samples[:,j])))
        diagnostics.append(dict(block_hours=hours,blocks=nblocks,nonempty_blocks=int((counts>0).sum()),
                                last_block_hours=float((DEV_STOP-FIRST)/pd.Timedelta(hours=1)-(nblocks-1)*hours),
                                n=n,seed=seed,draw_sha256=__import__('hashlib').sha256(draws.astype('<i8').tobytes()).hexdigest(),
                                maxT_critical=critical,candidates_in_maxT=REDUCED))
    return pd.DataFrame(intervals),pd.concat(all_samples,ignore_index=True),diagnostics


def choose(summary, intervals):
    rows=[]
    for variant in REDUCED:
        tests=intervals.loc[intervals.variant.eq(variant)]
        assert set(tests.block_hours)=={168,672}
        passed=bool(tests.nominal_noninferior.all() and tests.simultaneous_noninferior.all())
        metrics=summary.loc[summary.variant.eq(variant)].iloc[0].to_dict()
        rows.append({**metrics,"qualified":passed,"adaptive":variant.startswith("C_top"),
                     "worst_nominal_low":float(tests.ap_low.min()),"worst_simultaneous_low":float(tests.ap_simultaneous_low.min())})
    candidates=pd.DataFrame(rows).sort_values(["n_features_max","n_features_mean","mean_monthly_ap","brier","variant"],ascending=[True,True,False,True,True])
    good=candidates.loc[candidates.qualified]
    primary=good.iloc[0].variant if len(good) else "B"
    alternative=None
    if primary!="B":
        opposite=good.loc[good.adaptive.ne(primary.startswith("C_top"))]
        if len(opposite): alternative=opposite.iloc[0].variant
    selection=dict(primary=primary,sensitivity=alternative,
                   reduced_selected=primary!="B",margin=.01,
                   rule="both nominal and simultaneous lower AP limits > -0.01 for both 168 and 672 h; registered complexity order",
                   qualification_is_conditional=True,final_evaluation_executed=False,
                   frozen_fixed_list_independently_validated=False,
                   brier_used_as_tiebreaker_not_noninferiority_endpoint=True)
    return candidates,selection


def stability():
    history=json.loads((HERE/"selection_history.json").read_text())
    sets={(r["variant"],r["origin"]):set(r["features"]) for r in history}
    origins=sorted({r["origin"] for r in history})
    rows=[];freq=[]
    for variant in ["C_top5","C_top10","C_top20"]:
        for i,left in enumerate(origins):
            for right in origins[i+1:]:
                a,b=sets[(variant,left)],sets[(variant,right)]
                rows.append(dict(variant=variant,left=left,right=right,jaccard=len(a&b)/len(a|b),
                                 intersection=len(a&b),union=len(a|b),adjacent=origins.index(right)==i+1))
        for feature in FEATURES:
            count=sum(feature in sets[(variant,o)] for o in origins)
            freq.append(dict(variant=variant,feature=feature,selected_origins=count,fraction=count/len(origins)))
    save_frame(pd.DataFrame(rows),"subset_stability.csv")
    save_frame(pd.DataFrame(freq),"selection_frequency.csv")


def main():
    check_protocol();start=time.monotonic()
    pred=pd.read_parquet(HERE/"predictions.parquet")
    assert pred.time_local.max()<DEV_STOP and pred.origin.max()=="2026-03"
    aligned=strictly_align(pred)
    ledger=pd.DataFrame(json.loads((HERE/"model_ledger.json").read_text()))
    summary,monthly,deltas=summaries(pred,ledger)
    save_frame(summary,"summary.csv");save_frame(monthly,"monthly_metrics.csv");save_frame(deltas,"monthly_contrasts.csv")
    save_frame(historical_checks(aligned),"historical_anchor_checks.csv")
    intervals,samples,diagnostics=bootstrap(aligned)
    save_frame(intervals,"paired_intervals.csv");save_frame(samples,"bootstrap_replicates.parquet")
    write_json(HERE/"bootstrap_diagnostics.json",diagnostics)
    candidates,selection=choose(summary,intervals)
    save_frame(candidates,"qualification.csv");write_json(HERE/"selection_decision.json",selection)
    stability()
    write_json(HERE/"provenance/analysis_execution.json",dict(
        utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        protocol_sha256=sha(HERE/"protocol.json"),code_sha256=sha(HERE/"analyze.py"),
        selection=selection))
    print(json.dumps(selection,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
