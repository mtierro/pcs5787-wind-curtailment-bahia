"""Explicit future reproduction of fixed H2 systems; never invoked by validation.

This release provides the recipe and dry-run checks. No model was refitted while
preparing the publication. Thresholds here are the saved historical thresholds,
not a new optimization; see experimental threshold provenance for their history.
"""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
from .release_tools import read_data,verify_manifest,CAL,KEY,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True,type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
    d1,d2,_,_,base=read_data(a.data_root);verify_manifest(d1);verify_manifest(d2)
    registry=json.loads((d2/'models/model_registry.json').read_text())
    membership=pd.read_parquet(d2/'evaluation/split_membership.parquet')
    history=pd.read_parquet(d2/'evaluation/predictions.parquet')
    plans=[]
    for r in registry:
        rows=membership[(membership.origin==r['origin'])&(membership.role=='train')].sort_values('membership_position')
        tr=rows[KEY].merge(base,on=KEY,validate='one_to_one').sort_values(['time_local','region']).reset_index(drop=True)
        O=pd.Timestamp(r['origin']+'-01',tz='America/Bahia');stop=O-pd.Timedelta(hours=169)
        assert len(tr)==r['n_train'] and tr.y_ENE_02.notna().all() and tr.sp_wind.notna().all()
        assert tr.time_local.ge(stop-pd.DateOffset(months=3)).all() and tr.time_local.max()==O-pd.Timedelta(hours=170)
        plans.append({'origin':r['origin'],'variant':r['variant'],'n_train':len(tr),'first':str(tr.time_local.min()),'last':str(tr.time_local.max()),'features':r['meteorological_columns']})
        if a.dry_run:continue
        if a.output is None:ap.error('--output is required without --dry-run')
        if not hasattr(a,'_created'):
            a.output.mkdir(parents=True,exist_ok=False);a._created=True
        import joblib,lightgbm
        from .v2.learners import make_model
        assert lightgbm.__version__=='4.7.0'
        params=r['parameters'];spec={'algorithm':'lightgbm','calendar':'strong','seed':5787,'params':{
          'leaves':params['num_leaves'],'min_leaf':params['min_child_samples'],'iterations':params['n_estimators'],
          'learning_rate':params['learning_rate'],'l2':params['reg_lambda'],'colsample':params['colsample_bytree']}}
        model=make_model(spec,r['meteorological_columns'])
        model.fit(tr,tr.y_ENE_02.to_numpy(int),model__categorical_feature=list(range(6)))
        assert model.named_steps['model'].get_params()==params
        p=a.output/r['origin']/(r['variant']+'.joblib');p.parent.mkdir(exist_ok=True);joblib.dump(model,p,compress=3)
        g=history[(history.origin==r['origin'])&(history.variant==r['variant'])]
        x=g[KEY].merge(base,on=KEY,validate='one_to_one');z=model.named_steps['representation'].transform(x)
        prob=model.named_steps['model'].booster_.predict(z,num_threads=4)
        g=g.copy();g['p_reproduced']=prob;g.to_parquet(p.with_suffix('.parquet'),index=False)
        plans[-1]['max_probability_error']=float(np.max(np.abs(prob-g.p)))
    if a.output is not None and not a.dry_run:(a.output/'training_receipt.json').write_text(json.dumps(plans,indent=2)+'\n')
    print(json.dumps({'dry_run':a.dry_run,'models':len(plans),'plans':plans},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
