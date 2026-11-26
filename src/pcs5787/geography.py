"""Geografia auditável: CEG normalizado, composição ONS e centróides SIGA."""
import numpy as np
import pandas as pd
from .common import ROOT, SOURCE, CFG, save_json, sha256


def normalize_ceg(s):
    return s.astype('string').str.strip().str.replace(r'\.(\d+)$',lambda m:'.'+str(int(m[1])),regex=True)


def distance_km(lat1,lon1,lat2,lon2):
    a,b,c,d=map(np.radians,[lat1,lon1,lat2,lon2])
    return 6371*2*np.arcsin(np.sqrt(np.clip(np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2,0,1)))


def main():
    s=pd.read_csv(ROOT/'data/raw/siga/20261012/siga.csv',sep=';',decimal=',')
    s=s[(s.SigUFPrincipal=='BA')&(s.SigTipoGeracao=='EOL')].copy()
    s['ceg_norm']=normalize_ceg(s.CodCEG)
    assert not s.ceg_norm.duplicated().any()
    m=pd.read_csv(ROOT/'metadata/membership_monthly.csv')
    m['ceg_norm']=normalize_ceg(m.ceg)
    m=m.merge(s[['ceg_norm','CodCEG','NomEmpreendimento','NumCoordNEmpreendimento','NumCoordEEmpreendimento','MdaPotenciaFiscalizadaKw','DatGeracaoConjuntoDados']],on='ceg_norm',how='left',validate='many_to_one')
    assert m.NumCoordNEmpreendimento.notna().all() and m.NumCoordEEmpreendimento.notna().all()
    assert m.NumCoordNEmpreendimento.between(-19,-8).all() and m.NumCoordEEmpreendimento.between(-47,-37).all()
    m[['ceg','ceg_norm','CodCEG','nom_usina','NomEmpreendimento','NumCoordNEmpreendimento','NumCoordEEmpreendimento','MdaPotenciaFiscalizadaKw']].drop_duplicates().to_csv(ROOT/'metadata/plant_coordinates.csv',index=False)
    a=pd.read_csv(ROOT/'reports/tables/inmet_station_year.csv')
    d=a[a.year.le(2024)]
    e=d.groupby('station').agg(wind_min=('coverage_vento_ms','min'),gust_min=('coverage_rajada_ms','min'),years=('year','nunique'),latitude=('latitude','median'),longitude=('longitude','median'),name=('name','first'))
    shifts=d.merge(e[['latitude','longitude']],left_on='station',right_index=True,suffixes=('','_anchor'))
    shifts['km']=distance_km(shifts.latitude,shifts.longitude,shifts.latitude_anchor,shifts.longitude_anchor)
    e['max_dev_displacement_km']=shifts.groupby('station').km.max()
    minimum=CFG['weather_station_min_development_coverage']
    e['eligible']=e.wind_min.ge(minimum)&e.gust_min.ge(minimum)&e.years.eq(3)&e.max_dev_displacement_km.le(CFG['station_max_displacement_km'])
    e.to_csv(ROOT/'metadata/station_selection.csv')
    anchors=e[e.eligible].sort_index()
    # Identidade regional é a área de proximidade de âncoras fixas; não um nome inferido.
    g=m.drop_duplicates(['id_ons_conjuntousina','period','ceg_norm']).groupby(['id_ons_conjuntousina','period']).agg(latitude=('NumCoordNEmpreendimento','mean'),longitude=('NumCoordEEmpreendimento','mean'),n_plants=('ceg_norm','nunique'),capacity_mw=('MdaPotenciaFiscalizadaKw',lambda x:x.sum()/1000)).reset_index().rename(columns={'id_ons_conjuntousina':'id_ons'})
    dist=distance_km(g.latitude.to_numpy()[:,None],g.longitude.to_numpy()[:,None],anchors.latitude.to_numpy()[None,:],anchors.longitude.to_numpy()[None,:])
    nearest=dist.argmin(axis=1)
    g['region']=anchors.index[nearest]
    g['station_distance_km']=dist.min(axis=1)
    g['spatial_eligible']=g.station_distance_km.le(CFG['station_max_distance_km'])
    g['valid_from']=pd.to_datetime(g.period.str.replace('_','-')+'-01')
    g['valid_to_exclusive']=g.valid_from+pd.offsets.MonthBegin(1)
    g.to_csv(ROOT/'metadata/unit_region_monthly.csv',index=False)
    save_json('metadata/geography_provenance.json',{'siga_url':'https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel','siga_sha256':sha256(ROOT/'data/raw/siga/20261012/siga.csv'),'siga_snapshot':s.DatGeracaoConjuntoDados.unique().tolist(),'license':'ODbL; licença integrada não presumida','n_matched_plants':m.ceg.nunique(),'n_anchor_stations':len(anchors),'regions':sorted(g.region.unique()),'maximum_centroid_distance_km':g.station_distance_km.max(),'rule':'CEG só normaliza zeros à esquerda da revisão final; centróide não ponderado das usinas observadas por conjunto/mês; âncoras fixadas apenas em 2022–2024; raio máximo 200 km','limitation':'Coordenadas SIGA atuais aproximadas, sem comprovação de estabilidade histórica. Composição mensal é retrospectiva e pode mudar dentro do mês; não se reparte potência entre regiões. Sensibilidades de distância e composição exigidas.'})
    print(g.groupby('region').agg(unit_months=('id_ons','size'),max_km=('station_distance_km','max')).to_string())


if __name__=='__main__':main()
