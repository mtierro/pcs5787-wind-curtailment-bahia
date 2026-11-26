"""Catálogo NoSQL documental para proveniência e qualidade; observações em Parquet."""
import json
import pandas as pd
from tinydb import TinyDB, Query
from .common import ROOT, save_json


def main():
    with TinyDB(ROOT/'data/processed/catalog_nosql.json') as db:
        db.drop_tables()
        for folder,typ in [('metadata','metadata'),('reports/tables','result')]:
            for p in sorted((ROOT/folder).glob('*.csv')):
                db.table('artifacts').insert({'path':str(p.relative_to(ROOT)),'type':typ,'bytes':p.stat().st_size})
        a=pd.read_csv(ROOT/'reports/tables/inmet_station_year.csv')
        db.table('station_years').insert_multiple(json.loads(a.to_json(orient='records')))
        q=Query();result=db.table('station_years').search((q.year==2024)&(q.coverage_vento_ms>=.9))
        save_json('reports/tables/nosql_query.json',{'tool':'TinyDB','collection':'station_years','query':'year == 2024 AND coverage_vento_ms >= 0.9','matching_station_codes':[r['station'] for r in result],'count':len(result),'purpose':'catálogo documental; medições em Parquet'})


if __name__=='__main__':main()
