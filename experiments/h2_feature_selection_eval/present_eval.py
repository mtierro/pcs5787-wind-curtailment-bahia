"""Three publication figures and review tables; never fit or select."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pcs5787-final-selection-matplotlib')
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_eval import HERE,DEV,sha,save,write_json,registered
from eval_core import VARIANTS,REDUCED,START,STOP,columns,load_base

COLORS={'A':'#777777','B':'#2b4c72','C_top5':'#b44932','C_spatial':'#237d78'}


def figure_save(fig,name):
    folder=HERE/'figures';folder.mkdir(exist_ok=True)
    for ext in ['pdf','png']:fig.savefig(folder/f'{name}.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)


def tables():
    summary=pd.read_csv(HERE/'summary.csv');intervals=pd.read_csv(HERE/'absolute_intervals.csv')
    main=summary.copy()
    for hours in [168,672]:
        x=intervals[intervals.block_hours.eq(hours)].set_index('variant')
        for metric in ['ap','brier','f1']:
            for end in ['low','high']:main[f'{metric}_{hours}h_{end}']=main.variant.map(x[f'{metric}_{end}'])
    save(main,'main_table.csv')
    old=pd.read_csv(DEV/'summary.csv');rows=[]
    for v in VARIANTS:
        a=old.loc[old.variant.eq(v)].iloc[0];b=summary.loc[summary.variant.eq(v)].iloc[0]
        rows.append(dict(variant=v,development_n=int(a.n),evaluation_n=int(b.n),development_prevalence=a.prevalence,
                         evaluation_prevalence=b.prevalence,development_ap=a.ap,evaluation_ap=b.ap,
                         development_brier=a.brier,evaluation_brier=b.brier,
                         interpretation='development adaptive top-k versus evaluation fixed list' if v=='C_top5' else 'same representation; different time/population',
                         threshold_difference='historical F3 warm start in final A/B; monthly OOF in development' if v in ['A','B'] else 'frozen monthly OOF warm start; then own available evaluation OOF'))
    save(pd.DataFrame(rows),'development_vs_evaluation.csv')
    base=load_base();base=base.loc[base.time_local.ge(START)&base.time_local.lt(STOP)]
    exclusions=[]
    for month,g in base.groupby(base.time_local.dt.strftime('%Y-%m')):
        label=g.y_ENE_02.notna();sp=g.sp_wind.notna()
        exclusions.append(dict(origin=month,total_grid=len(g),eligible=int((label&sp).sum()),
                               label_missing_only=int((~label&sp).sum()),sp_wind_missing_only=int((label&~sp).sum()),
                               both_missing=int((~label&~sp).sum())))
    save(pd.DataFrame(exclusions),'eligibility_audit.csv')
    records=json.loads((HERE/'reduced_models.json').read_text());cost=[]
    for v in REDUCED:
        r=[x for x in records if x['variant']==v]
        cost.append(dict(variant=v,models=6,n_features=len(columns(v)),n_effective=6+2*len(columns(v)),
                         fit_seconds_total=sum(x['fit_seconds'] for x in r),fit_seconds_mean=np.mean([x['fit_seconds'] for x in r]),
                         prediction_ms_per_1000=np.mean([x['predict_seconds']*1e6/x['n_eval'] for x in r]),
                         serialized_bytes=sum((HERE/x['model_path']).stat().st_size for x in r)))
    save(pd.DataFrame(cost),'computational_cost.csv')
    counts=[]
    for r in records:
        model=joblib.load(HERE/r['model_path']);k=len(columns(r['variant']))
        imp=model.named_steps['model'].booster_.feature_importance('split')
        counts.append(dict(variant=r['variant'],origin=r['origin'],calendar_splits=int(imp[:6].sum()),
                           meteorological_value_splits=int(imp[6:6+k].sum()),missing_indicator_splits=int(imp[6+k:].sum())))
    save(pd.DataFrame(counts),'split_counts.csv')


def figures():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    s=pd.read_csv(HERE/'summary.csv').set_index('variant')
    fig,ax=plt.subplots(figsize=(7.7,4.8),layout='constrained')
    for v in VARIANTS:
        r=s.loc[v];ax.scatter(r.n_meteorological,r.ap,s=85,c=COLORS[v],zorder=3)
        shift={'A':(9,8),'B':(-13,-35),'C_top5':(7,9),'C_spatial':(8,-33)}[v]
        ax.annotate(f'{v}: AP={r.ap:.6f}\n{int(r.n_effective)} entradas efetivas',(r.n_meteorological,r.ap),
                    xytext=shift,textcoords='offset points',fontsize=9,ha='right' if v=='B' else 'left')
    ax.set(xlim=(-3,57),ylim=(.884,.906),xticks=[0,5,14,51],xlabel='Número de variáveis meteorológicas',
           ylabel='Average Precision agrupada',title='Avaliação retrospectiva: listas fixas, abril–setembro/2026')
    ax.grid(alpha=.18);figure_save(fig,'figure1_AP_complexity')
    monthly=pd.read_csv(HERE/'monthly_contrasts.csv');origins=sorted(monthly.origin.unique())
    fig,ax=plt.subplots(figsize=(8.3,4.4),layout='constrained')
    for v in REDUCED:
        g=monthly[monthly.contrast.eq(v+'_minus_B')].set_index('origin').loc[origins]
        ax.plot(range(6),g.delta_ap*100,marker='o',c=COLORS[v],label=v)
        other='C_spatial' if v=='C_top5' else 'C_top5'
        other_values=monthly[monthly.contrast.eq(other+'_minus_B')].set_index('origin').loc[origins,'delta_ap'].to_numpy()*100
        for i,val in enumerate(g.delta_ap*100):
            ax.annotate(f'{val:+.2f}',(i,val),xytext=(0,8 if val>=other_values[i] else -16),textcoords='offset points',ha='center',fontsize=8)
    ax.axhline(0,c='grey',lw=.8);ax.grid(alpha=.18)
    ax.set(xticks=range(6),xticklabels=['Abr','Mai','Jun','Jul','Ago','Set'],ylim=(-.42,.80),
           xlabel='Origem mensal de 2026',ylabel='AP(reduzido) − AP(B), pontos percentuais',
           title='Variação mensal; os seis meses não são réplicas independentes')
    ax.legend();figure_save(fig,'figure2_monthly_AP_differences')
    intervals=pd.read_csv(HERE/'paired_intervals.csv').set_index(['contrast','block_hours'])
    names=['C_top5_minus_B','C_spatial_minus_B','B_minus_A','C_top5_minus_A','C_spatial_minus_A']
    labels=['C_top5 − B (principal)','C_spatial − B (sensibilidade)','B − A','C_top5 − A','C_spatial − A']
    fig,ax=plt.subplots(figsize=(9,5.4),layout='constrained')
    for hours,dy,color in [(168,-.14,'#237d78'),(672,.14,'#b44932')]:
        for i,c in enumerate(names):
            r=intervals.loc[(c,hours)];y=i+dy
            ax.plot([r.ap_low*100,r.ap_high*100],[y,y],c=color,lw=2)
            ax.scatter(r.delta_ap*100,y,c=color,s=35,label=f'IC nominal 95% — blocos {hours} h' if i==0 else None)
            if np.isfinite(r.ap_simultaneous_low):ax.scatter(r.ap_simultaneous_low*100,y,c=color,marker='<',s=40,label='Limite inferior simultâneo (2 contrastes)' if i==0 and hours==168 else None)
    ax.axvline(-1,c='black',ls='--',lw=1,label='Margem de não inferioridade: −1 p.p.')
    ax.axvline(0,c='grey',lw=.8);ax.grid(axis='x',alpha=.18)
    ax.set(yticks=range(5),yticklabels=labels,xlabel='Diferença de AP, pontos percentuais',
           title='Contrastes pareados; 2.000 réplicas por tamanho de bloco')
    ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,fontsize=8)
    figure_save(fig,'figure3_paired_AP_intervals')
    write_json(HERE/'provenance/figures.json',dict(source_sha256=sha(HERE/'present_eval.py'),matplotlib=matplotlib.__version__,
                count=3,formats=['pdf','png'],selection_modified=False))


if __name__=='__main__':registered();tables();figures();print('Three figures and review tables written')
