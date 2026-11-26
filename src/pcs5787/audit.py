"""Inventário, hashes e auditoria sem alteração do acervo importado."""
import base64
from collections import Counter
from io import BytesIO
import json
import re
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import ROOT, SOURCE, sha256, save_json

WEATHER_COLS = {
    "vento_ms": "VENTO, VELOCIDADE HORARIA (m/s)",
    "rajada_ms": "VENTO, RAJADA MAXIMA (m/s)",
    "temperatura_c": "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
    "umidade_pct": "UMIDADE RELATIVA DO AR, HORARIA (%)",
    "radiacao_kjm2": "RADIACAO GLOBAL (Kj/m²)",
    "precipitacao_mm": "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)",
}


def parse_numeric(series):
    s = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce").astype(float).replace([-9999, 9999], np.nan)


def parse_weather(df, station):
    time = pd.to_datetime(df["Data"].str.replace("/", "-", regex=False) + " " +
                          df["Hora UTC"].str[:2] + ":00:00", utc=True, errors="raise")
    out = pd.DataFrame({"time_utc": time, "station": station})
    for name, col in WEATHER_COLS.items():
        out[name] = parse_numeric(df[col])
    # Apenas impossibilidades/sentinelas; extremos plausíveis são preservados.
    for name in ["vento_ms", "rajada_ms", "precipitacao_mm"]:
        out.loc[out[name] < 0, name] = np.nan
    out.loc[~out.umidade_pct.between(0, 100), "umidade_pct"] = np.nan
    return out


def inventory():
    files = sorted(p for p in SOURCE.rglob("*") if p.is_file())
    rows = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
             "sha256": sha256(p)} for p in files]
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "metadata/manifest_sha256.csv", index=False)
    expected = pd.read_csv(SOURCE / "metadata/manifests/manifest_downloads_20261008.csv")
    hashes = {p.name: sha for p, sha in zip(files, frame.sha256)}
    checks = []
    for _, row in expected.iterrows():
        name = row.get("nome_arquivo_armazenado")
        name = name if pd.notna(name) else row["nome_arquivo"]
        h = row.get("sha256_armazenado")
        h = h if pd.notna(h) else row["sha256"]
        checks.append({"file": name, "present": name in hashes, "match": hashes.get(name) == h})
    pd.DataFrame(checks).to_csv(ROOT / "reports/tables/hash_conferencia_downloads.csv", index=False)
    schemas = []
    for p in files:
        if p.suffix == ".parquet":
            meta = pq.read_metadata(p)
            schemas.append({"path": str(p.relative_to(ROOT)), "rows": meta.num_rows,
                            "columns": meta.num_columns, "schema": str(meta.schema.to_arrow_schema().remove_metadata())})
    pd.DataFrame(schemas).to_csv(ROOT / "metadata/schemas.csv", index=False)
    nb = json.loads((ROOT / "referencias_codex/referencias/grupo/notebook_preliminar_grupo9.ipynb").read_text())
    save_json("metadata/notebook_preliminar_auditoria.json", {
        "cells": len(nb["cells"]), "reexecuted": False,
        "note": "Código/saídas inspecionados; dependência de Colab/Drive substituída por módulos locais. Saídas persistidas não certificam reprodução.",
        "cells_inventory": [{"index": i, "type": c["cell_type"], "execution_count": c.get("execution_count"),
                             "saved_outputs": len(c.get("outputs", [])), "source_characters": len("".join(c["source"]))}
                            for i, c in enumerate(nb["cells"])]})
    print(f"Inventário: {len(files)} arquivos; {len(checks)} hashes de downloads confrontados", flush=True)


def weather_audit():
    all_frames, stats, source_checks = [], [], []
    catalog = pd.read_csv(SOURCE / "metadata/dictionaries/cadastro_estacoes_inmet_bahia_20261008.csv")
    for year in range(2022, 2027):
        zpath = SOURCE / f"data/raw/inmet/20261008/INMET_{year}.zip"
        with zipfile.ZipFile(zpath) as z:
            members = {p.rsplit("/", 1)[-1]: p for p in z.namelist() if "_BA_" in p.upper()}
            for p in sorted((SOURCE / f"data/interim/inmet_bahia/{year}").glob("*.parquet")):
                raw = pd.read_parquet(p)
                station = re.search(r"_(A\d{3})_", p.name).group(1)
                meta = pq.read_metadata(p).metadata
                source_name = meta[b"source_csv_name"].decode()
                blob = z.read(members[source_name])
                orig = pd.read_csv(BytesIO(blob), sep=";", skiprows=8, encoding="latin1", dtype="string", keep_default_na=False)
                orig = orig[[c for c in orig if not c.startswith("Unnamed:")]]
                identical = orig.astype("string").equals(raw.astype("string"))
                assert identical, p
                header_match = base64.b64decode(meta[b"inmet_header_raw_latin1_base64"]) == b"".join(blob.splitlines(keepends=True)[:8])
                assert header_match
                source_checks.append({"file": p.name, "zip_member": members[source_name], "table_equal": identical, "header_equal": header_match})
                frame = parse_weather(raw, station)
                assert not frame.duplicated(["station", "time_utc"]).any(), p
                start = pd.Timestamp(f"{year}-01-01", tz="UTC")
                end = pd.Timestamp(f"{year+1}-01-01" if year < 2026 else "2026-10-01", tz="UTC")
                expected = int((end-start).total_seconds()/3600)
                assert frame.time_utc.between(start, end, inclusive="left").all()
                cat = catalog[(catalog.codigo_wmo == station) & (catalog.ano == year)].iloc[0]
                row = {"station": station, "name": cat.estacao, "year": year, "rows": len(frame),
                       "expected_hours": expected, "missing_rows": expected-len(frame),
                       "latitude": cat.latitude, "longitude": cat.longitude,
                       "first": frame.time_utc.min(), "last": frame.time_utc.max()}
                for var in WEATHER_COLS:
                    row[f"valid_{var}"] = int(frame[var].notna().sum())
                    row[f"coverage_{var}"] = row[f"valid_{var}"]/expected
                row["gust_below_wind"] = int((frame.rajada_ms < frame.vento_ms).sum())
                stats.append(row)
                all_frames.append(frame)
        print(f"INMET {year}: {sum(x['year']==year for x in stats)} estações verificadas contra ZIP", flush=True)
    data = pd.concat(all_frames, ignore_index=True).sort_values(["station", "time_utc"])
    assert not data.duplicated(["station", "time_utc"]).any()
    data.to_parquet(ROOT / "data/interim/weather.parquet", index=False)
    pd.DataFrame(stats).to_csv(ROOT / "reports/tables/inmet_station_year.csv", index=False)
    pd.DataFrame(source_checks).to_csv(ROOT / "reports/tables/inmet_raw_verification.csv", index=False)


def ons_audit():
    stats, memberships, coverage, frames, comparisons = [], [], [], [], []
    for p in sorted((SOURCE / "data/interim/ons_bahia").rglob("*.parquet")):
        a = pd.read_parquet(p)
        bpath = SOURCE / f"data/interim/ons_detalhamento_bahia/{p.parent.name}/{p.name.replace('ONS_BA_', 'ONS_DETAIL_BA_')}"
        b = pd.read_parquet(bpath)
        assert (a.id_estado == "BA").all() and (b.id_estado == "BA").all()
        period = p.stem[-7:]
        duplicate_a = int(a.duplicated(["id_ons", "din_instante"]).sum())
        duplicate_b = int(b.duplicated(["ceg", "din_instante"]).sum())
        exact_duplicate_b = int(b.duplicated().sum())
        assert duplicate_a == 0, (p, duplicate_a)
        # Remover somente repetições integralmente idênticas; conflitos abortam.
        b = b.drop_duplicates()
        assert not b.duplicated(["ceg", "din_instante"]).any(), (bpath, duplicate_b)
        a["time_local"] = pd.to_datetime(a.din_instante).dt.tz_localize("America/Bahia")
        assert a.time_local.dt.minute.isin([0, 30]).all()
        assert a.time_local.dt.second.eq(0).all()
        restricted = a.val_geracaolimitada.notna()
        formula = (a.val_geracaoreferencia - a.val_geracao).clip(lower=0)
        delta = (formula[restricted] - a.loc[restricted, "val_geracaonaorealizadaapurada"]).abs()
        mins = a[["num_minutos_rel", "num_minutos_cnf", "num_minutos_ene"]]
        row = {"period": period, "rows": len(a), "units": a.id_ons.nunique(),
               "detail_rows": len(b), "detail_plants": b.ceg.nunique(),
               "duplicate_aggregate_key": duplicate_a, "duplicate_detail_key": duplicate_b,
               "exact_detail_duplicates_removed": exact_duplicate_b,
               "reference_missing": int(a.val_geracaoreferencia.isna().sum()),
               "reference_final_missing": int(a.val_geracaoreferenciafinal.isna().sum()),
               "reason_without_limit": int((a.cod_razaorestricao.notna() & ~restricted).sum()),
               "limit_without_reason": int((a.cod_razaorestricao.isna() & restricted).sum()),
               "restricted_missing_gnr": int((restricted & a.val_geracaonaorealizadaapurada.isna()).sum()),
               "formula_discrepancy_gt_0_002": int((delta > .002).sum()),
               "multiple_reasons_minutes": int((mins.gt(0).sum(axis=1)>1).sum()),
               "negative_power": int(a.select_dtypes('number').filter(like='val_').lt(0).any(axis=1).sum()),
               "minutes_outside_0_30": int(mins.lt(0).any(axis=1).sum()+mins.gt(30).any(axis=1).sum())}
        for reason, count in a.cod_razaorestricao.fillna("SEM_CODIGO").value_counts().items():
            row[f"reason_{reason}"] = int(count)
        stats.append(row)
        comp = b.groupby(["id_ons_conjuntousina", "din_instante"]).val_geracaoverificada.sum().rename("detail_generation")
        comp.index = comp.index.set_names(["id_ons", "din_instante"])
        comp = a.set_index(["id_ons", "din_instante"])[["val_geracao"]].join(comp, validate="one_to_one")
        assert len(comp) == len(a)
        comparisons.append({"period": period, "matched": int(comp.detail_generation.notna().sum()),
                            "aggregate_rows": len(comp), "max_abs_mw": float((comp.val_geracao-comp.detail_generation).abs().max()),
                            "mean_abs_mw": float((comp.val_geracao-comp.detail_generation).abs().mean())})
        m = b.groupby(["id_ons_conjuntousina", "ceg", "id_ons", "nom_usina"]).din_instante.agg(["min", "max", "count"]).reset_index()
        m["period"] = period
        memberships.append(m)
        c = a.groupby("id_ons").time_local.agg(["min", "max", "count"]).reset_index()
        c["period"] = period
        c["missing_between_first_last"] = ((c["max"]-c["min"]).dt.total_seconds()/1800 + 1 - c["count"]).astype(int)
        coverage.append(c)
        keep = ["time_local", "id_ons", "nom_usina", "val_geracao", "val_geracaoreferencia", "val_geracaonaorealizadaapurada", "val_geracaolimitada", "cod_razaorestricao", "num_minutos_rel", "num_minutos_cnf", "num_minutos_ene", "num_minutos_restricao"]
        frames.append(a[keep])
        if p.stem.endswith('_12') or p.stem.endswith('2026_09'):
            print(f"ONS até {period}: chaves agregadas e CEG×instante verificadas", flush=True)
    data = pd.concat(frames, ignore_index=True)
    assert not data.duplicated(["id_ons", "time_local"]).any()
    data.to_parquet(ROOT / "data/interim/ons.parquet", index=False)
    pd.DataFrame(stats).fillna(0).to_csv(ROOT / "reports/tables/ons_monthly_audit.csv", index=False)
    pd.concat(memberships).to_csv(ROOT / "metadata/membership_monthly.csv", index=False)
    pd.concat(coverage).to_csv(ROOT / "reports/tables/ons_unit_coverage.csv", index=False)
    pd.DataFrame(comparisons).to_csv(ROOT / "reports/tables/ons_detail_conservation.csv", index=False)


def main():
    inventory()
    weather_audit()
    ons_audit()
    print("Auditoria estrutural concluída.", flush=True)


if __name__ == "__main__":
    main()
