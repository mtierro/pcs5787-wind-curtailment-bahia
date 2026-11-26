"""Development-only training campaign. No code path evaluates April–September."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import resource
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from audit import HERE, ROOT, sha, digest, write_json, verify, check_protocol
from selection_core import (CONFIG, FEATURES, CALENDAR, FAMILIES, VARIANTS, FIRST, FIRST_TRAIN,
                            FREEZE, DEV_STOP, ORIGINS, bounds, cutoff, columns_for, ranking_order,
                            fit_model, predict, threshold_from_history, score, frame_fingerprint, strictly_align)


def save_frame(frame, name):
    path = HERE/name
    assert path.resolve().is_relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet": frame.to_parquet(path, index=False)
    else: frame.to_csv(path, index=False)


def load_development():
    check_protocol()
    assert importlib.metadata.version("lightgbm") == "4.7.0"
    frozen = json.loads((ROOT/"configs/v2/finalists_frozen.json").read_text())
    assert sha(ROOT/"data/processed/v2/analytical.parquet") == frozen["dataset_sha256"]
    assert sha(ROOT/"configs/v2/feature_groups.json") == frozen["feature_groups_sha256"]
    for name, expected in frozen["engine_sha256"].items():
        assert sha(ROOT/"src/pcs5787/v2"/name) == expected
    definitions = pd.read_csv(ROOT/"reports/h2_feature_importance/feature_definitions.csv")
    assert definitions.feature.tolist() == FEATURES
    assert definitions.latest_physical_lag_hours.ge(2).all()
    filters = [("time_local", ">=", FIRST_TRAIN), ("time_local", "<", DEV_STOP)]
    cols = ["time_local", "y_ENE_02"]+CALENDAR+FEATURES
    raw = pd.read_parquet(ROOT/"data/processed/v2/analytical.parquet", columns=cols, filters=filters)
    assert raw.time_local.min() == FIRST_TRAIN and raw.time_local.max() == DEV_STOP-pd.Timedelta(hours=1)
    assert not raw.duplicated(["region", "time_local"]).any()
    assert raw.groupby("time_local").size().eq(8).all()
    assert len(raw) == int((DEV_STOP-FIRST_TRAIN)/pd.Timedelta(hours=1))*8
    assert not np.isinf(raw[FEATURES].to_numpy()).any()
    base = raw.loc[raw.y_ENE_02.notna() & raw.sp_wind.notna()].sort_values(["time_local", "region"]).reset_index(drop=True)
    base["row_id"] = np.arange(len(base))
    assert set(base.y_ENE_02.unique()) == {0,1}
    return base, dict(source="data/processed/v2/analytical.parquet", columns=cols,
                      filters=[(name, op, str(value)) for name, op, value in filters],
                      materialized_first=str(raw.time_local.min()), materialized_last=str(raw.time_local.max()),
                      materialized_rows=len(raw), eligible_rows=len(base),
                      latest_label_available=str(base.time_local.max()+pd.Timedelta(hours=169)),
                      final_period_rows_materialized=0, prior_importance_rankings_read=False,
                      final_models_loaded=False, source_sha256=frozen["dataset_sha256"])


def split_frame(base, origin):
    b = bounds(origin)
    train = base.loc[base.time_local.ge(b["train_start"]) & base.time_local.lt(b["train_stop"])]
    valid = base.loc[base.time_local.ge(b["validation_start"]) & base.time_local.lt(b["validation_stop"])]
    return train, valid, b


def membership(frame, origin, role):
    return pd.DataFrame(dict(origin=origin, role=role, row_id=frame.row_id.to_numpy()))


def date_record(frame, origin, role):
    return dict(origin=origin, role=role, n=len(frame), positives=int(frame.y_ENE_02.sum()),
                first=str(frame.time_local.min()), last=str(frame.time_local.max()),
                latest_label_available=str(frame.time_local.max()+pd.Timedelta(hours=169)),
                membership_sha256=frame_fingerprint(frame))


def save_model(model, relative):
    path = HERE/"models"/relative
    assert path.resolve().is_relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return str(path.relative_to(HERE)), sha(path)


def inner_ranking(train, origin, repeat=False):
    b = bounds(origin); label = origin.strftime("%Y-%m")
    inner = train.loc[train.time_local.lt(b["inner_stop"])]
    holdout = train.loc[train.time_local.ge(b["rank_origin"]) & train.time_local.lt(b["rank_stop"])]
    assert len(inner) and len(holdout)
    assert set(inner.row_id).isdisjoint(holdout.row_id)
    assert inner.time_local.max()+pd.Timedelta(hours=169) <= b["rank_origin"]-pd.Timedelta(hours=1)
    assert holdout.time_local.max()+pd.Timedelta(hours=169) <= origin-pd.Timedelta(hours=1)
    model, seconds, caught = fit_model(inner, FEATURES, "B", b["rank_origin"], "inner")
    p, z, _ = predict(model, holdout, FEATURES)
    booster = model.named_steps["model"].booster_
    phi = booster.predict(z, pred_contrib=True, num_threads=4)
    raw = booster.predict(z, raw_score=True, num_threads=4)
    assert phi.shape == (len(holdout),109)
    margin_residual = float(np.max(np.abs(phi.sum(axis=1)-raw)))
    probability_residual = float(np.max(np.abs(expit(phi.sum(axis=1))-p)))
    assert margin_residual <= CONFIG["validation_tolerances"]["shap_margin"]
    assert probability_residual <= CONFIG["validation_tolerances"]["shap_probability"]
    joint = phi[:,6:57]+phi[:,57:108]
    importance = np.abs(joint).mean(axis=0)
    rank = {j:i+1 for i,j in enumerate(ranking_order(importance))}
    table = pd.DataFrame([dict(origin=label, feature=name, original_index=j, rank=rank[j],
            mean_abs_joint=float(importance[j]), mean_abs_value=float(np.abs(phi[:,6+j]).mean()),
            mean_abs_flag=float(np.abs(phi[:,57+j]).mean()),
            missing_fraction=float(np.isnan(z[:,6+j]).mean()), n_rank=len(holdout)) for j,name in enumerate(FEATURES)])
    model_path, model_hash = save_model(model, f"inner_rank_{label}.joblib")
    record = dict(origin=label, purpose="freeze_ranking_only" if origin==FREEZE else "development_ranking",
                  rank_origin=str(b["rank_origin"]), train_first=str(inner.time_local.min()), train_last=str(inner.time_local.max()),
                  rank_first=str(holdout.time_local.min()), rank_last=str(holdout.time_local.max()),
                  n_train=len(inner), n_rank=len(holdout), train_hash=frame_fingerprint(inner), rank_hash=frame_fingerprint(holdout),
                  fit_seconds=seconds, model_path=model_path, model_sha256=model_hash,
                  shap_margin_residual=margin_residual, shap_probability_residual=probability_residual,
                  warnings=caught, parameters=model.named_steps["model"].get_params())
    if repeat:
        other, extra_seconds, _ = fit_model(inner, FEATURES, "B", b["rank_origin"], "repeat_inner")
        repeated = other.named_steps["model"].booster_.predict(other.named_steps["representation"].transform(holdout), pred_contrib=True, num_threads=4)
        assert np.array_equal(phi, repeated)
        record.update(repeat_shap_max_residual=float(np.max(np.abs(phi-repeated))), repeat_fit_seconds=extra_seconds)
    return importance, table, record, inner, holdout


def run():
    verify()
    if (HERE/"predictions.parquet").exists():
        raise RuntimeError("Campaign outputs already exist. Use a new experimental package to preserve this run.")
    start = time.monotonic()
    base, access = load_development()
    save_frame(base[["row_id", "time_local", "region", "y_ENE_02"]], "development_population.parquet")
    write_json(HERE/"provenance/data_access.json", access)
    history = {variant:None for variant in VARIANTS}
    predictions, ledgers, selection_rows, ranks, rank_records, members, dates = [], [], [], [], [], [], []
    for origin in ORIGINS:
        label = origin.strftime("%Y-%m")
        train, valid, b = split_frame(base, origin)
        assert len(valid) and valid.time_local.max() < DEV_STOP
        importance, ranking, rank_record, inner, holdout = inner_ranking(train, origin, repeat=origin==FIRST)
        ranks.append(ranking); rank_records.append(rank_record)
        for role, frame in [("outer_train",train), ("outer_validation",valid), ("inner_fit",inner), ("inner_rank",holdout)]:
            members.append(membership(frame,label,role)); dates.append(date_record(frame,label,role))
        train_hash = frame_fingerprint(train); valid_hash = frame_fingerprint(valid)
        for variant in VARIANTS:
            cols = columns_for(variant, importance); flags = variant != "B_no_indicators"
            threshold, threshold_record = threshold_from_history(history[variant], origin)
            model, seconds, caught = fit_model(train, cols, variant, origin, "outer")
            p, z, prediction_seconds = predict(model, valid, cols, flags)
            frame = valid[["row_id", "time_local", "region"]].copy()
            frame["y"] = valid.y_ENE_02.to_numpy(int); frame["p"] = p
            frame["threshold"] = threshold; frame["origin"] = label; frame["variant"] = variant
            model_path, model_hash = save_model(model, f"{label}/{variant}.joblib")
            rec = dict(origin=label, variant=variant, n_features=len(cols), effective_entries=z.shape[1],
                       flags=flags, n_train=len(train), n_valid=len(valid), train_hash=train_hash, valid_hash=valid_hash,
                       train_first=str(train.time_local.min()), train_last=str(train.time_local.max()),
                       validation_first=str(valid.time_local.min()), validation_last=str(valid.time_local.max()),
                       fit_seconds=seconds, predict_seconds=prediction_seconds,
                       predict_ms_per_1000=prediction_seconds*1e6/len(valid),
                       model_path=model_path, model_sha256=model_hash,
                       month_unknown=int(np.isnan(z[:,3]).sum()), region_month_unknown=int(np.isnan(z[:,5]).sum()),
                       threshold=threshold, threshold_history=threshold_record, warnings=caught,
                       parameters=model.named_steps["model"].get_params(),
                       booster_parameters=model.named_steps["model"].booster_.params,
                       encoder_categories=[values.tolist() for values in model.named_steps["representation"].encoder_.categories_])
            if origin==FIRST and variant in ["A", "B", "C_top5", "B_no_indicators"]:
                repeated, extra_seconds, _ = fit_model(train, cols, variant, origin, "repeat_outer")
                repeat_p, _, _ = predict(repeated, valid, cols, flags)
                assert np.array_equal(p, repeat_p)
                rec.update(repeat_probability_max_residual=float(np.max(np.abs(p-repeat_p))), repeat_fit_seconds=extra_seconds)
            ledgers.append(rec); predictions.append(frame)
            selection_rows.append(dict(origin=label, variant=variant, features=cols,
                                       missing_indicators=[f"missing::{c}" for c in cols] if flags else [],
                                       n_features=len(cols), effective_entries=z.shape[1], selection_sha256=digest(cols)))
            history[variant] = frame if history[variant] is None else pd.concat([history[variant],frame],ignore_index=True)
        print(f"{label}: {len(train)} train / {len(valid)} validation; all 11 variants completed", flush=True)
    pred = pd.concat(predictions,ignore_index=True)
    aligned = strictly_align(pred)
    assert len(aligned["B"]) == 85185
    assert pred.time_local.max() < DEV_STOP and pred.origin.max() == "2026-03"
    save_frame(pred,"predictions.parquet")
    save_frame(pd.concat(members,ignore_index=True),"split_membership.parquet")
    save_frame(pd.DataFrame(dates),"split_dates.csv")
    save_frame(pd.concat(ranks,ignore_index=True),"inner_rankings.csv")
    write_json(HERE/"model_ledger.json",ledgers)
    write_json(HERE/"selection_history.json",selection_rows)
    write_json(HERE/"inner_model_ledger.json",rank_records)
    write_json(HERE/"provenance/campaign_execution.json",dict(
        utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        outer_models=165, inner_models=15, repeat_fits=5, final_evaluation_executed=False,
        code_sha256={name:sha(HERE/name) for name in ["selection_core.py","campaign.py","audit.py"]},
        protocol_sha256=sha(HERE/"protocol.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["run"])
    parser.parse_args()
    run()
