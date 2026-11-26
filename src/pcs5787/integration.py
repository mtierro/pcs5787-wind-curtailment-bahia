"""Base região×hora, com limites de atribuição dos motivos e cobertura explícita."""
import numpy as np
import pandas as pd
from .common import ROOT, save_json
from .features import historical_features, calendar_features
from .geography import distance_km

REASONS=['ENE','CNF','REL']


def energy_bounds(x):
    """Potência média do patamar × 0,5 h; não multiplicar novamente pela duração.

    Motivos mistos: atribuição não identificada, intervalo [0, GNR] por motivo.
    Negativo só é definido em registro presente, sem limitação e sem minutos.
    """
    out=pd.DataFrame(index=x.index)
    restricted=x.val_geracaolimitada.notna()
    minutes=x[[f'num_minutos_{r.lower()}' for r in REASONS]]
    active=minutes.gt(0)
    n_active=active.sum(axis=1)
    no_restriction=~restricted & x.cod_razaorestricao.isna() & x.val_geracaonaorealizadaapurada.isna() & minutes.fillna(0).eq(0).all(axis=1) & x.num_minutos_restricao.fillna(0).eq(0)
    valid_restricted=restricted & x.cod_razaorestricao.isin(REASONS) & x.val_geracaolimitada.ge(0) & x.val_geracaonaorealizadaapurada.ge(0) & minutes.notna().all(axis=1) & minutes.ge(0).all(axis=1) & minutes.le(30).all(axis=1) & x.num_minutos_restricao.between(1,30) & minutes.sum(axis=1).eq(x.num_minutos_restricao) & n_active.ge(1)
    valid=(no_restriction|valid_restricted)&x.val_geracao.ge(0)&x.val_geracaoreferencia.ge(0)
    out['valid_row']=valid
    out['reference_mwh']=x.val_geracaoreferencia*.5
    out['generation_mwh']=x.val_geracao*.5
    cut=x.val_geracaonaorealizadaapurada.where(restricted,0)*.5
    out['cut_total_mwh']=cut.where(valid)
    for reason in REASONS:
        positive=active[f'num_minutos_{reason.lower()}']
        out[f'cut_low_{reason}']=cut.where(positive & n_active.eq(1),0).where(valid)
        out[f'cut_high_{reason}']=cut.where(positive,0).where(valid)
    out['mixed_row']=n_active.gt(1)
    return out


def label_bounds(frame, reason, tau):
    lower=frame[f'cut_low_{reason}']/frame.reference_mwh
    upper=frame[f'cut_high_{reason}']/frame.reference_mwh
    valid=frame.ons_eligible & frame.reference_mwh.gt(0)
    label=pd.Series(np.nan,index=frame.index)
    label.loc[valid & upper.le(tau)]=0
    label.loc[valid & lower.gt(tau)]=1
    return label


def main():
    x=pd.read_parquet(ROOT/'data/interim/ons.parquet')
    x['period']=x.time_local.dt.strftime('%Y_%m')
    geo=pd.read_csv(ROOT/'metadata/unit_region_monthly.csv')
    x=x.merge(geo,on=['id_ons','period'],how='left',validate='many_to_one')
    assert x.region.notna().all()
    assert not x.duplicated(['id_ons','time_local']).any()
    x['hour']=x.time_local.dt.floor('h')
    e=energy_bounds(x)
    x=pd.concat([x,e],axis=1)
    # O universo é a frota registrada, não toda a capacidade territorial.
    x['unit_hour_count']=x.groupby(['id_ons','hour']).id_ons.transform('size')
    x['valid_row'] &= x.unit_hour_count.eq(2)&x.spatial_eligible
    x['capacity_suspicious']=(x.capacity_mw.gt(0)&(x.val_geracao.gt(1.5*x.capacity_mw)|x.val_geracaoreferencia.gt(1.5*x.capacity_mw)))
    x['valid_row'] &= ~x.capacity_suspicious
    agg={'reference_mwh':('reference_mwh','sum'),'generation_mwh':('generation_mwh','sum'),'cut_total_mwh':('cut_total_mwh','sum'), 'rows':('id_ons','size'),'units':('id_ons','nunique'),'valid_rows':('valid_row','sum'),'mixed_rows':('mixed_row','sum'),'max_distance_km':('station_distance_km','max'),'capacity_suspicious_rows':('capacity_suspicious','sum')}
    for r in REASONS:
        for bound in ['low','high']:agg[f'cut_{bound}_{r}']=(f'cut_{bound}_{r}','sum')
    h=x.groupby(['region','hour']).agg(**agg).reset_index()
    h['ons_eligible']=h.valid_rows.eq(h.rows)&h.rows.eq(2*h.units)&h.reference_mwh.gt(0)
    # Impede interpretar agregação parcial como energia total da população.
    for col in ['reference_mwh','generation_mwh','cut_total_mwh']+[f'cut_{b}_{r}' for r in REASONS for b in ['low','high']]:
        h.loc[~h.ons_eligible,col]=np.nan
    good=x[x.valid_row]
    good_hours=h[h.ons_eligible][['region','hour']]
    recon=good.merge(good_hours,on=['region','hour'],validate='many_to_one')
    np.testing.assert_allclose(h.reference_mwh.sum(),recon.reference_mwh.sum(),rtol=1e-12)
    np.testing.assert_allclose(h.cut_total_mwh.sum(),recon.cut_total_mwh.sum(),rtol=1e-12)
    stations=pd.read_csv(ROOT/'metadata/station_selection.csv').set_index('station')
    coverage=pd.read_csv(ROOT/'reports/tables/inmet_station_year.csv')
    weather=pd.read_parquet(ROOT/'data/interim/weather.parquet')
    frames=[]
    for region in sorted(h.region.unique()):
        w=weather[weather.station.eq(region)].set_index('time_utc').sort_index()
        w.index=w.index.tz_convert('America/Bahia')
        anchor=stations.loc[region]
        for row in coverage[coverage.station.eq(region)].itertuples():
            if distance_km(row.latitude,row.longitude,anchor.latitude,anchor.longitude)>5:
                w.loc[w.index.tz_convert('UTC').year==row.year,['vento_ms','rajada_ms']]=np.nan
        base=h[h.region.eq(region)].set_index('hour').sort_index()
        grid=pd.date_range('2022-01-01','2026-10-01',inclusive='left',freq='h',tz='America/Bahia')
        base=base.reindex(grid)
        base['region']=region
        base['ons_eligible']=base.ons_eligible.fillna(False).astype(bool)
        base=base.join(calendar_features(grid))
        for delay in [1,2,6]:
            f=historical_features(w,delay_hours=delay)
            if delay!=2:f=f.add_suffix(f'_delay{delay}')
            base=base.join(f)
        base=base.join(w[['vento_ms','rajada_ms']],how='left')
        frames.append(base.rename_axis('time_local').reset_index())
    base=pd.concat(frames,ignore_index=True)
    met=[f'wind_lag{l}' for l in [1,3,6,24]]+[f'gust_lag{l}' for l in [1,24]]+['wind_mean24','gust_max24']
    base['weather_eligible']=base[met].notna().all(axis=1)
    base['year']=base.time_local.dt.year
    for tau in [.01,.02,.05]:
        for reason in REASONS:base[f'y_{reason}_{int(tau*100):02d}']=label_bounds(base,reason,tau)
    assert not base.duplicated(['region','time_local']).any()
    base.to_parquet(ROOT/'data/processed/regional_hourly.parquet',index=False)
    base.groupby(['region','year']).agg(hours=('region','size'),ons_eligible=('ons_eligible','sum'),weather_eligible=('weather_eligible','sum')).to_csv(ROOT/'reports/tables/regional_coverage.csv')
    x.loc[~x.valid_row,['id_ons','time_local','region','capacity_suspicious','unit_hour_count']].to_parquet(ROOT/'data/interim/ons_exclusions.parquet',index=False)
    prevalence=[]
    for tau in [1,2,5]:
        for r in REASONS:
            for yr in [2022,2023,2024]:
                d=base[base.year.eq(yr)&base.weather_eligible];y=d[f'y_{r}_{tau:02d}'].dropna()
                prevalence.append({'tau_pct':tau,'reason':r,'year':yr,'n':len(y),'positives':y.sum(),'prevalence':y.mean()})
    pd.DataFrame(prevalence).to_csv(ROOT/'reports/tables/development_tau.csv',index=False)
    save_json('reports/tables/integration_summary.json',{'rows':len(base),'regions':base.region.nunique(),'ons_valid_hours':int(base.ons_eligible.sum()),'weather_valid_hours':int(base.weather_eligible.sum()),'invalid_ons_rows':int((~x.valid_row).sum()),'capacity_suspicious_rows':int(x.capacity_suspicious.sum()),'primary_weather_delay_hours':2,'conservation_reference_mwh':float(h.reference_mwh.sum()),'conservation_cut_mwh':float(h.cut_total_mwh.sum()),'event_attribution':'limites conservadores por motivo; mistos podem gerar rótulo desconhecido','scope':'frota observada ONS por região/mês; ausência de registro fora do ciclo observado não garante ausência de corte'})
    print(pd.DataFrame(prevalence).to_string(index=False))


if __name__=='__main__':main()
