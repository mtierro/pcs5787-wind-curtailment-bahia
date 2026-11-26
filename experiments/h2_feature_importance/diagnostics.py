"""Input definitions, structural use and deterministic matrix fingerprints."""
import hashlib
import json
import re
import numpy as np
import pandas as pd
from run_analysis import HERE, ROOT, CONFIG, FEATURES, load_context
from provenance import sha, json_write


def definitions():
    inv = pd.read_csv(HERE / "feature_inventory.csv")
    rows = []
    for row in inv.to_dict("records"):
        name = row["feature"]
        source = "estação âncora; vento=vento_ms; rajada=rajada_ms"
        function = "src/pcs5787/v2/data.py:lagged_weather"
        latest = 2
        minimum = "operandos necessários válidos; senão NaN"
        operation = ""
        if re.fullmatch(r"(wind|gust)_lag\d+", name):
            lag = int(re.search(r"\d+", name).group())
            latest = lag+1
            operation = f"{'vento' if name.startswith('wind') else 'rajada'}(t-{latest}h); delay_hours=2"
            function = "src/pcs5787/features.py:historical_features"
        elif name in ["wind_mean24", "gust_max24"]:
            operation = ("média de vento" if name == "wind_mean24" else "máximo de rajada") + " entre t-25h e t-2h"
            minimum = "24 de 24 observações"
            function = "src/pcs5787/features.py:historical_features"
        elif name.startswith("wind_ramp"):
            window = int(re.search(r"\d+", name).group())
            operation = f"vento(t-2h)-vento(t-{window+2}h)"
        elif name == "wind_cubic_proxy": operation = "vento(t-2h)^3; proxy, sem unidade de potência"
        elif name == "gust_excess": operation = "rajada(t-2h)-vento(t-2h)"
        elif name == "gust_ratio": operation = "rajada(t-2h)/(vento(t-2h)+1 m/s)"
        elif name.startswith("wind_coverage"):
            window = int(re.search(r"\d+", name).group())
            operation = f"fração de ventos válidos entre t-{window+1}h e t-2h"
            minimum = f"grade completa de {window} horas; ausentes contam como zero"
        elif name.endswith("_partial"):
            stat, window = re.match(r"wind_(mean|std|min|max)(\d+)_partial", name).groups()
            window = int(window)
            operation = f"{stat} de vento entre t-{window+1}h e t-2h; std usa ddof=1"
            minimum = f"mínimo {max(3, window//2)} de {window} valores válidos"
        elif name.endswith("_lag2"):
            stem = name[:-5]
            operations = {"wind_u": "-vento*sin(direção)", "wind_v": "-vento*cos(direção)",
                          "direction_sin": "sin(direção)", "direction_cos": "cos(direção)"}
            operation = operations.get(stem, stem) + " em t-2h"
            if stem in operations: operation += "; ângulo convertido de graus para radianos"
            source = "weather_extended; estação âncora"
        elif name.startswith("sp_"):
            function = "src/pcs5787/v2/data.py:spatial_weather"
            source = "até 5 estações a até 200 km; geometria anterior a 2025; pesos 1/max(distância_km,10)^2"
            operations = {
                "sp_wind": "vento médio ponderado em t-2h",
                "sp_count": "número de estações com vento válido em t-2h",
                "sp_distance": "distância média ponderada das estações com vento válido em t-2h",
                "sp_wind_std": "desvio espacial ponderado populacional do vento em t-2h",
                "sp_wind_q25": "quantil espacial 0,25 não ponderado do vento em t-2h",
                "sp_wind_q75": "quantil espacial 0,75 não ponderado do vento em t-2h",
                "sp_gust": "rajada média ponderada em t-2h",
                "sp_u": "componente -vento*sin(direção), média ponderada em t-2h",
                "sp_v": "componente -vento*cos(direção), média ponderada em t-2h",
                "sp_temperature": "temperatura média ponderada em t-2h",
                "sp_ramp6": "sp_wind(t)-sp_wind(t-6h), medidas físicas t-2h e t-8h",
                "sp_mean24": "média temporal de sp_wind; medidas físicas entre t-25h e t-2h",
                "sp_std24": "desvio temporal de sp_wind, ddof=1; medidas físicas entre t-25h e t-2h",
                "sp_gust_excess": "sp_gust(t)-sp_wind(t), medidas físicas em t-2h"}
            operation = operations[name]
            minimum = "pesos renormalizados por variável e disponibilidade; zero estações -> NaN, exceto sp_count=0"
            if name in ["sp_mean24", "sp_std24"]: minimum += "; mínimo 12 de 24 horas"
        assert operation
        rows.append({**row, "source_series": source, "construction_function": function,
                     "operation": operation, "latest_physical_lag_hours": latest,
                     "construction_missing_rule": minimum})
    pd.DataFrame(rows).to_csv(HERE / "feature_definitions.csv", index=False)


def main():
    definitions()
    frozen = json.loads((ROOT / "configs/v2/finalists_frozen.json").read_text())
    assert sha(ROOT / "data/processed/v2/analytical.parquet") == frozen["dataset_sha256"]
    assert sha(ROOT / "configs/v2/feature_groups.json") == frozen["feature_groups_sha256"]
    for name, digest in frozen["engine_sha256"].items():
        assert sha(ROOT / "src/pcs5787/v2" / name) == digest
    for split in CONFIG["splits"]:
        v = pd.read_csv(HERE / split / "shap_variables.csv")
        v["abs_contribution_on_missing"] = v.mean_abs_joint_missing.fillna(0)*v.missing_fraction
        v["abs_contribution_on_present"] = v.mean_abs_joint_present.fillna(0)*(1-v.missing_fraction)
        v[["scope", "group", "n", "feature", "family", "missing_fraction", "abs_contribution_on_missing",
           "abs_contribution_on_present"]].to_csv(HERE / split / "shap_availability.csv", index=False)
        contexts, obs, _, _, _ = load_context(split)
        entries, fingerprints, bases = [], [], []
        for c in contexts:
            counts = c["booster"].feature_importance(importance_type="split")
            gains = c["booster"].feature_importance(importance_type="gain")
            cols = pd.read_csv(HERE / "effective_columns.csv")
            for row in cols.itertuples():
                entries.append(dict(origin=c["origin"], column=row.column, feature=row.name,
                                    component=row.component, splits=int(counts[row.column]), gain=float(gains[row.column])))
            fingerprints.append(dict(origin=c["origin"], n=len(c["z"]), columns=108,
                                     matrix_sha256=hashlib.sha256(c["z"].astype("<f8").tobytes()).hexdigest(),
                                     labels_sha256=hashlib.sha256(c["obs"].y.to_numpy(dtype="<i8").tobytes()).hexdigest(),
                                     times_utc_ns_sha256=hashlib.sha256(pd.DatetimeIndex(c["obs"].time_local).as_unit("ns").asi8.astype("<i8").tobytes()).hexdigest()))
        pd.DataFrame(entries).to_csv(HERE / split / "model_structure.csv", index=False)
        pd.DataFrame(fingerprints).to_csv(HERE / split / "matrix_fingerprints.csv", index=False)
    json_write(HERE / "provenance/definition_checks.json", dict(
        frozen_dataset_hash_matches=True, frozen_feature_group_hash_matches=True,
        frozen_engine_hashes_match=True, native_feature_names="Column_0 ... Column_107",
        effective_column_aliases="research labels only; binary model unchanged",
        source_clock="historical measured times; no independent publication timestamp validation"))


if __name__ == "__main__": main()
