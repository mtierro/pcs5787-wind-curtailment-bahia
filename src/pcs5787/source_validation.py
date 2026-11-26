"""Conferências adicionais do ONS, sem repetir inventário ou EDA já concluídos."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, SOURCE


def validate_raw():
    target = ROOT/'reports/tables/ons_raw_verification.csv'
    rows = pd.read_csv(target).to_dict('records') if target.exists() else []
    done = {r['file'] for r in rows if r['equal']}
    for folder, rawfolder, prefix, rawprefix in [
        ('ons_bahia','ons','ONS_BA_','RESTRICAO_COFF_EOLICA_'),
        ('ons_detalhamento_bahia','ons_detalhamento','ONS_DETAIL_BA_','RESTRICAO_COFF_EOLICA_DETAIL_')]:
        for p in sorted((SOURCE/f'data/interim/{folder}').rglob('*.parquet')):
            if p.name in done: continue
            period = p.stem.replace(prefix,'')
            paths = list((SOURCE/f'data/raw/{rawfolder}/20261008/{period[:4]}').glob(f'*{period}.csv.gz'))
            assert len(paths)==1, (period,paths)
            raw = pd.concat([c[c.id_estado.eq('BA')] for c in pd.read_csv(paths[0],sep=';',dtype='string',chunksize=150000)],ignore_index=True)
            interim = pd.read_parquet(p).reset_index(drop=True)
            assert list(raw.columns)==list(interim.columns) and raw.shape==interim.shape, p
            for col in interim:
                if pd.api.types.is_numeric_dtype(interim[col]):
                    np.testing.assert_allclose(pd.to_numeric(raw[col],errors='raise').to_numpy(dtype=float,na_value=np.nan),interim[col].to_numpy(dtype=float),rtol=1e-12,atol=1e-10,equal_nan=True,err_msg=f'{p.name}:{col}')
                else:
                    assert raw[col].astype('string').fillna('').equals(interim[col].astype('string').fillna('')), (p,col)
            rows.append({'file':p.name,'raw_file':str(paths[0].relative_to(ROOT)),'rows':len(raw),'columns':len(raw.columns),'equal':True})
            pd.DataFrame(rows).to_csv(target,index=False)
            print(f'{len(rows)}/114 {p.name}: igualdade confirmada',flush=True)


def discrepancies():
    details=[]
    for p in sorted((SOURCE/'data/interim/ons_bahia').rglob('*.parquet')):
        a=pd.read_parquet(p)
        b=pd.read_parquet(SOURCE/f'data/interim/ons_detalhamento_bahia/{p.parent.name}/{p.name.replace("ONS_BA_","ONS_DETAIL_BA_")}').drop_duplicates()
        g=b.groupby(['id_ons_conjuntousina','din_instante']).agg(detail_generation=('val_geracaoverificada','sum'),n_plants=('ceg','nunique')).reset_index().rename(columns={'id_ons_conjuntousina':'id_ons'})
        c=a.merge(g,on=['id_ons','din_instante'],how='left',validate='one_to_one')
        c['delta_mw']=c.val_geracao-c.detail_generation
        flag=c.delta_mw.abs().gt(1)|c.detail_generation.isna()
        cols=['id_ons','nom_usina','din_instante','val_geracao','detail_generation','delta_mw','n_plants']
        details.append(c.loc[flag,cols])
    out=pd.concat(details,ignore_index=True)
    out.to_parquet(ROOT/'data/interim/ons_detail_discrepancies.parquet',index=False)
    out.groupby('id_ons').agg(n=('delta_mw','size'),max_abs_mw=('delta_mw',lambda x:x.abs().max()),median_abs_mw=('delta_mw',lambda x:x.abs().median())).to_csv(ROOT/'reports/tables/ons_discrepancies_by_unit.csv')
    out.loc[out.delta_mw.abs().nlargest(30).index].to_csv(ROOT/'reports/tables/ons_discrepancies_top.csv',index=False)


if __name__=='__main__':
    import sys
    if '--discrepancies' in sys.argv: discrepancies()
    else: validate_raw()
