"""Compact development tables and publication figures; no fitting or selection."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pcs5787-selection-matplotlib")
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit import HERE, sha, write_json, check_protocol
from selection_core import FEATURES, FAMILIES, VARIANTS, score
from campaign import save_frame


def save(fig, name):
    folder=HERE/"figures";folder.mkdir(exist_ok=True)
    for ext in ["pdf","png"]:
        fig.savefig(folder/f"{name}.{ext}",dpi=200,bbox_inches="tight")
    plt.close(fig)


def extra_tables(pred, monthly, ranks, ledger):
    regional=[]
    for (variant,region),frame in pred.groupby(["variant","region"]):
        regional.append(dict(variant=variant,region=region,**score(frame.y,frame.p,frame.threshold)))
    save_frame(pd.DataFrame(regional),"regional_metrics.csv")
    family={f:name for name,cols in FAMILIES.items() for f in cols}
    rows=[]
    for feature,group in ranks.groupby("feature"):
        rows.append(dict(feature=feature,family=family[feature],
                         mean_abs_joint_weighted=np.average(group.mean_abs_joint,weights=group.n_rank),
                         mean_abs_value_weighted=np.average(group.mean_abs_value,weights=group.n_rank),
                         mean_abs_flag_weighted=np.average(group.mean_abs_flag,weights=group.n_rank),
                         mean_rank=group["rank"].mean(),min_rank=group["rank"].min(),max_rank=group["rank"].max()))
    importance=pd.DataFrame(rows).sort_values(["mean_abs_joint_weighted","feature"],ascending=[False,True])
    save_frame(importance,"development_importance_summary.csv")
    families=importance.groupby("family")[["mean_abs_joint_weighted","mean_abs_value_weighted","mean_abs_flag_weighted"]].sum().reset_index()
    families["share_sum_individual_importance"]=families.mean_abs_joint_weighted/families.mean_abs_joint_weighted.sum()
    # This is a sum of individual logical-variable scores, NOT mean(abs(sum(phi_family))).
    save_frame(families,"development_family_sum_importance.csv")
    deltas=pd.read_csv(HERE/"monthly_contrasts.csv")
    stability=[]
    for variant,g in deltas.groupby("variant"):
        stability.append(dict(variant=variant,mean_delta_ap=g.delta_ap.mean(),min_delta_ap=g.delta_ap.min(),
                              worst_origin=g.loc[g.delta_ap.idxmin(),"origin"],max_delta_ap=g.delta_ap.max(),
                              best_origin=g.loc[g.delta_ap.idxmax(),"origin"],months_ap_above_B=int(g.delta_ap.gt(0).sum()),
                              months_ap_below_B=int(g.delta_ap.lt(0).sum()),
                              months_delta_below_minus_001=int(g.delta_ap.lt(-.01).sum()),
                              mean_delta_brier=g.delta_brier.mean(),max_delta_brier=g.delta_brier.max()))
    save_frame(pd.DataFrame(stability),"temporal_stability.csv")
    checks=[]
    for row in ledger:
        if row["variant"]!="B":continue
        model=joblib.load(HERE/row["model_path"])
        splits=model.named_steps["model"].booster_.feature_importance("split")
        checks.append(dict(origin=row["origin"],value_splits=int(splits[6:57].sum()),
                           flag_splits=int(splits[57:108].sum()),calendar_splits=int(splits[:6].sum())))
    save_frame(pd.DataFrame(checks),"missing_indicator_splits.csv")
    return importance,deltas


def figures(summary,intervals,importance,deltas):
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"savefig.facecolor":"white"})
    colors={v:("#b34a36" if v=="C_top5" else "#287b76" if v=="C_spatial" else "#2c4d71" if v=="B" else "#6c7279") for v in VARIANTS}
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    offsets={"A":(4,8),"B":(-10,-16),"C_spatial":(-20,9),"C_top5":(5,-15),"C_top10":(5,-13),
             "C_top20":(4,6),"B_minus_anchor":(-74,8),"B_minus_temporal":(-38,9),
             "B_minus_atmospheric":(-62,-17),"B_minus_spatial":(-40,9),"B_no_indicators":(-84,10)}
    for ax,metric,label in zip(axes,["ap","brier"],["AP agrupada (maior é melhor)","Brier agrupado (menor é melhor)"]):
        for row in summary.itertuples():
            x=row.n_features_mean;y=getattr(row,metric)
            ax.scatter(x,y,c=colors[row.variant],marker="s" if row.variant=="B_no_indicators" else "o",s=38,zorder=3)
            shift=offsets[row.variant]
            if metric=="brier" and row.variant=="B_no_indicators":shift=(-87,-15)
            ax.annotate(row.variant,(x,y),xytext=shift,textcoords="offset points",fontsize=7)
        ax.set(xlabel="Número de variáveis meteorológicas (média entre origens)",ylabel=label,xlim=(-3,56))
        ax.grid(alpha=.18)
    fig.suptitle("Desenvolvimento: 85.185 previsões por variante; top-k adaptativo")
    save(fig,"complexity_performance")
    names=["C_top5","C_top10","C_spatial","C_top20","B_minus_anchor","B_minus_temporal","B_minus_atmospheric","B_minus_spatial"]
    fig,ax=plt.subplots(figsize=(9,5.3),layout="constrained")
    for hours,dy,color in [(168,-.14,"#287b76"),(672,.14,"#b34a36")]:
        table=intervals.set_index(["variant","block_hours"])
        for i,v in enumerate(names):
            row=table.loc[(v,hours)]
            ax.plot([row.ap_low*100,row.ap_high*100],[i+dy]*2,color=color,lw=2)
            ax.scatter(row.delta_ap*100,i+dy,c=color,s=25,label=f"IC nominal 95%; blocos {hours} h" if i==0 else None)
            ax.scatter(row.ap_simultaneous_low*100,i+dy,c=color,marker="<",s=30,
                       label="Limite inferior simultâneo" if i==0 and hours==168 else None)
    ax.axvline(-1,c="black",ls="--",lw=1,label="Margem registrada: −1 p.p.")
    ax.axvline(0,c="grey",lw=.7)
    ax.set(yticks=range(len(names)),yticklabels=names,xlabel="AP(variante) − AP(B), pontos percentuais",
           title="Comparações pareadas condicionais; 2.000 réplicas por tamanho de bloco")
    ax.invert_yaxis();ax.grid(axis="x",alpha=.18)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.12),ncol=2,fontsize=7)
    save(fig,"paired_comparisons")
    names=[v for v in VARIANTS if v!="B"]
    matrix=deltas.pivot(index="variant",columns="origin",values="delta_ap").loc[names]*100
    fig,ax=plt.subplots(figsize=(12,5.2),layout="constrained")
    im=ax.imshow(matrix,cmap="RdBu",vmin=-9,vmax=9,aspect="auto")
    ax.set(xticks=range(len(matrix.columns)),xticklabels=matrix.columns,yticks=range(len(names)),yticklabels=names,
           title="Diferenças mensais de AP frente a B — origens dependentes, sem teste por mês")
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    for i in range(len(names)):
        for j in range(len(matrix.columns)):
            val=matrix.iloc[i,j]
            ax.text(j,i,f"{val:+.1f}",ha="center",va="center",fontsize=7,color="white" if abs(val)>5 else "black")
    fig.colorbar(im,ax=ax,label="ΔAP (pontos percentuais)",fraction=.03)
    save(fig,"monthly_contrasts")
    freq=pd.read_csv(HERE/"selection_frequency.csv").query("variant == 'C_top5'").set_index("feature")
    names=importance.head(15).feature.tolist()[::-1]
    imp=importance.set_index("feature").loc[names]
    fig,axes=plt.subplots(1,2,figsize=(10,6),sharey=True,layout="constrained",gridspec_kw={"width_ratios":[2,1]})
    axes[0].barh(names,imp.mean_abs_joint_weighted,color="#287b76")
    axes[0].set(xlabel="Média ponderada de |SHAP valor + SHAP ausência|\n(log-odds)",title="Referências internas, fora do ajuste")
    axes[1].barh(names,freq.loc[names,"selected_origins"],color="#b34a36")
    axes[1].set(xlim=(0,15.8),xticks=[0,5,10,15],xlabel="Origens com a variável selecionada",title="Política top-5 (15 origens)")
    for ax in axes:ax.grid(axis="x",alpha=.18)
    save(fig,"development_importance")


def main():
    check_protocol()
    pred=pd.read_parquet(HERE/"predictions.parquet")
    summary=pd.read_csv(HERE/"summary.csv");monthly=pd.read_csv(HERE/"monthly_metrics.csv")
    ranks=pd.read_csv(HERE/"inner_rankings.csv")
    ledger=json.loads((HERE/"model_ledger.json").read_text())
    importance,deltas=extra_tables(pred,monthly,ranks,ledger)
    figures(summary,pd.read_csv(HERE/"paired_intervals.csv"),importance,deltas)
    write_json(HERE/"provenance/figures.json",dict(code_sha256=sha(HERE/"present.py"),
              matplotlib=matplotlib.__version__,count_figures=4,formats=["pdf","png"],
              interpretation="development only; no new fitting or choice",importance_scale="raw log-odds"))
    print("Development diagnostic tables and four PDF/PNG figures written",flush=True)


if __name__=="__main__":main()
