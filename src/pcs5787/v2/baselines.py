"""Referência de frequências condicionais suavizadas; calibração/seleção temporal."""
import json
import time
import numpy as np
import pandas as pd
from .common import DATA,OUT,ROOT,REASONS,folds,population,training_mask,atomic_json,atomic_frame,digest,cutoff
from ..common import sha256
from .experiments import metrics


class CalendarFrequency:
    def __init__(self,alpha=50):self.alpha=alpha

    def fit(self,x,y):
        d=x[['region','hour','month']].copy();d['y']=np.asarray(y)
        self.prior_=(d.y.sum()+.5)/(len(d)+1)
        self.tables_=[]
        previous=np.full(len(d),self.prior_)
        for keys in [['region'],['region','hour'],['region','hour','month']]:
            d['prior']=previous
            z=d.groupby(keys).agg(count=('y','size'),total=('y','sum'),prior=('prior','mean'))
            z['p']=(z.total+self.alpha*z.prior)/(z['count']+self.alpha)
            self.tables_.append((keys,z.p))
            previous=d.set_index(keys).index.map(z.p).to_numpy(dtype=float)
        return self

    def predict(self,x):
        p=np.full(len(x),self.prior_)
        for keys,table in self.tables_:
            q=x.set_index(keys).index.map(table).to_numpy(dtype=float)
            p=np.where(np.isfinite(q),q,p)
        return p


def main():
    b=pd.read_parquet(DATA/'analytical.parquet').sort_values(['time_local','region'])
    data_hash=sha256(DATA/'analytical.parquet')
    rows=[];choices=[];predictions=[]
    for reason in REASONS:
        for pop in ['anchor_complete','spatial_recent']:
            eligible,target=population(b,reason,pop);all_results=[]
            for actual in ['window3','window6','window12','window24']:
                for alpha in [20,100]:
                    results=[];pieces=[]
                    for f in folds():
                        start=time.monotonic();tr=b[eligible&training_mask(b,f['origin'],actual)];va=b[eligible&b.time_local.ge(f['start'])&b.time_local.lt(f['stop'])]
                        m=CalendarFrequency(alpha).fit(tr,tr[target]);p=m.predict(va);score=metrics(va[target],p)
                        row={'run_id':digest([reason,pop,actual,alpha,str(f),data_hash,sha256(__file__)])[:20],'family':'frequency','question':'referência calendário suavizada','dataset_sha256':data_hash,'features':'region/hour/month','variant':'A','algorithm':'frequency','params':json.dumps({'alpha':alpha}),'seed':5787,'reason':reason,'population':pop,'strategy':actual,'alpha':alpha,'fold':f['name'],'n_train':len(tr),'train_start':str(tr.time_local.min()),'train_end':str(tr.time_local.max()),'status':'done','exclusion_reason':'',**score,'seconds':time.monotonic()-start};rows.append(row);results.append(score['ap'])
                        out=va[['time_local','region']].copy();out['y']=va[target].to_numpy();out['p']=p;out['fold']=f['name'];pieces.append(out)
                    all_results.append((np.mean(results),actual,alpha,pd.concat(pieces,ignore_index=True)))
            score,strategy,alpha,pred=max(all_results,key=lambda z:z[0]);choices.append({'reason':reason,'population':pop,'strategy':strategy,'alpha':alpha,'mean_ap':score})
            pred['reason']=reason;pred['population']=pop;predictions.append(pred)
    atomic_frame(pd.DataFrame(rows),OUT/'tables/frequency_validation.csv')
    atomic_frame(pd.concat(predictions,ignore_index=True),DATA/'frequency_oof.parquet')
    atomic_json(ROOT/'configs/v2/frequency_selected.json',choices)
    ledger_path=OUT/'tables/experiment_ledger.csv'
    ledger=pd.read_csv(ledger_path);extra=pd.DataFrame(rows)
    ledger=ledger[~ledger.run_id.isin(extra.run_id)]
    atomic_frame(pd.concat([ledger,extra],ignore_index=True),ledger_path)
    # Contagem/composição: padronização descritiva sem rótulos do externo.
    harmonization=[]
    for reason in REASONS:
        eligible,target=population(b,reason,'anchor_complete')
        d=b[eligible&b.time_local.lt(cutoff('2026-04-01'))&b.region.isin(['A408','A426','A429','A458'])&b.time_local.dt.month.le(3)&((b.time_local.dt.month<3)|(b.time_local.dt.day<25))].copy()
        d['cell']=d.region+'_'+d.hour.astype(str)+'_'+d.month.astype(str)
        z=d.groupby(['year','cell'])[target].agg(['sum','count']);rate=(z['sum']/z['count']).unstack(0);count=z['count'].unstack(0)
        for year0,year1 in [(2022,2025),(2024,2025),(2025,2026)]:
            cells=rate[[year0,year1]].dropna().index
            rates=rate.loc[cells,[year0,year1]];weights=count.loc[cells,[year0,year1]];weights=weights/weights.sum()
            raw0=(weights[year0]*rates[year0]).sum();raw1=(weights[year1]*rates[year1]).sum()
            within=((weights[year0]+weights[year1])/2*(rates[year1]-rates[year0])).sum()
            composition=((rates[year0]+rates[year1])/2*(weights[year1]-weights[year0])).sum()
            np.testing.assert_allclose(within+composition,raw1-raw0,atol=1e-12)
            harmonization.append(dict(reason=reason,year0=year0,year1=year1,cells=len(cells),rate0=raw0,rate1=raw1,total_change=raw1-raw0,within_stratum_component=within,composition_component=composition,scope='same four regions; Jan1-Mar24; shared region-hour-month cells; descriptive Kitagawa decomposition'))
    atomic_frame(pd.DataFrame(harmonization),OUT/'tables/prevalence_harmonized.csv')


if __name__=='__main__':main()
