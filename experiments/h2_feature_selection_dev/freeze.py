"""Freeze one final list from admissible development; never evaluate final months."""
import argparse
from datetime import datetime, timezone
import json
import time
import pandas as pd
from audit import HERE, ROOT, sha, digest, write_json, check_protocol
from selection_core import (CONFIG, FEATURES, CALENDAR, FREEZE, DEV_STOP, columns_for,
                            model_spec, threshold_from_history)
from campaign import load_development, split_frame, inner_ranking, membership, date_record, save_frame
from analyze import choose


def freeze():
    registration=check_protocol()
    path=HERE/"frozen/selection.json"
    if path.exists():raise RuntimeError("Frozen configuration already exists; use verify.")
    start=time.monotonic()
    summary=pd.read_csv(HERE/"summary.csv");intervals=pd.read_csv(HERE/"paired_intervals.csv")
    _,decision=choose(summary,intervals)
    assert decision==json.loads((HERE/"selection_decision.json").read_text())
    base,access=load_development()
    history,valid,b=split_frame(base,FREEZE)
    assert len(valid)==0, "No final validation population may be created"
    importance,ranking,rank_record,inner,holdout=inner_ranking(history,FREEZE)
    assert rank_record["purpose"]=="freeze_ranking_only"
    save_frame(ranking,"frozen/ranking_pre_april.csv")
    write_json(HERE/"frozen/selector_model.json",rank_record)
    save_frame(pd.concat([membership(history,"2026-04","freeze_history"),
                          membership(inner,"2026-04","inner_fit"),
                          membership(holdout,"2026-04","inner_rank")],ignore_index=True),"frozen/selection_membership.parquet")
    save_frame(pd.DataFrame([date_record(history,"2026-04","freeze_history"),date_record(inner,"2026-04","inner_fit"),
                            date_record(holdout,"2026-04","inner_rank")]),"frozen/selection_dates.csv")
    predictions=pd.read_parquet(HERE/"predictions.parquet")
    candidates=[decision["primary"]]
    if decision["sensitivity"]:candidates.append(decision["sensitivity"])
    outputs=[]
    for role,variant in zip(["primary","sensitivity"],candidates):
        columns=columns_for(variant,importance)
        effective=CALENDAR+columns+[f"missing::{c}" for c in columns]
        evidence=intervals.loc[intervals.variant.eq(variant)].to_dict("records") if variant!="B" else []
        outputs.append(dict(role=role,variant=variant,features=columns,
                            missing_indicators=[f"missing::{c}" for c in columns],
                            n_meteorological_features=len(columns),effective_columns=effective,
                            categorical_positions=list(range(6)),parameters=model_spec(variant),
                            effective_estimator_parameters={**rank_record["parameters"],"reg_lambda":40 if variant=="A" else 10},
                            feature_list_sha256=digest(columns),development_intervals=evidence,
                            validated_object="adaptive policy, not this fixed list" if variant.startswith("C_top") else "fixed structural representation",
                            final_period_performance="not evaluated"))
    initial_thresholds=[]
    for variant in dict.fromkeys(["A","B"]+candidates):
        threshold,record=threshold_from_history(predictions.loc[predictions.variant.eq(variant)],FREEZE)
        initial_thresholds.append(dict(variant=variant,threshold=threshold,history=record,
                                       source="new monthly development OOF; differs from original F1-F3 threshold history"))
    files=[HERE/"protocol.json",HERE/"summary.csv",HERE/"paired_intervals.csv",HERE/"selection_decision.json",
           HERE/"predictions.parquet",HERE/"frozen/ranking_pre_april.csv",HERE/rank_record["model_path"]]
    frozen=dict(version="2B-1-frozen-1",actual_created_at_utc=datetime.now(timezone.utc).isoformat(),
                logical_information_time="2026-03-31T23:00:00-03:00",
                development_stop_exclusive=str(DEV_STOP),selection_decision=decision,configurations=outputs,
                baseline_A=model_spec("A"),baseline_B=model_spec("B"),
                feature_definitions_path="reports/h2_feature_importance/feature_definitions.csv",
                feature_definitions_sha256=sha(ROOT/"reports/h2_feature_importance/feature_definitions.csv"),
                initial_april_thresholds=initial_thresholds,
                future_evaluation=dict(executed=False,authorization_required_in_next_mission=True,
                   period="2026-04-01 <= t < 2026-10-01",origins="monthly, April through September",
                   training_window="three calendar months ending at O-169h exclusive",
                   label_latency_hours=168,weather_latest_lag_hours=2,
                   population="y_ENE_02.notna() & sp_wind.notna(), shared by all variants",
                   list_policy="fixed for all six months; no new SHAP ranking, reselection or hyperparameter tuning",
                   thresholds="April values frozen here; later origins use variant's available prior 90-day OOF history, grid F1 .01..99, smallest tie, no probability calibration",
                   threshold_caveat="new development OOF warm start for all variants; report differences from historical A/B alert thresholds explicitly",
                   historical_baselines="verify A/B probabilities against original models; preserve original artifacts",
                   interpretation="retrospective additional analysis, not untouched confirmation"),
                dependencies=registration["dependencies"],python=registration["python"],
                historical_commit=registration["commit"],protocol_sha256=sha(HERE/"protocol.json"),
                source_hashes={p.name:sha(p) for p in sorted(HERE.glob("*.py"))},
                historical_source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/"src/pcs5787/v2/learners.py",ROOT/"src/pcs5787/v2/common.py",ROOT/"src/pcs5787/v2/calibration.py",ROOT/"src/pcs5787/v2/evaluation.py"]},
                artifacts_sha256={str(p.relative_to(HERE)):sha(p) for p in files},
                selection_model=rank_record,semantic_data_access=access)
    write_json(path,frozen)
    write_json(HERE/"frozen/configuration_manifest.json",dict(
        created_at_utc=frozen["actual_created_at_utc"],selection_sha256=sha(path),
        feature_lists={item["role"]:item["feature_list_sha256"] for item in outputs},
        protocol_sha256=sha(HERE/"protocol.json"),final_evaluation_executed=False))
    write_json(HERE/"provenance/freeze_execution.json",dict(seconds=time.monotonic()-start,
                additional_inner_fits=1,external_april_fits=0,final_period_predictions=0))
    print(json.dumps({"decision":decision,"lists":outputs},ensure_ascii=False,indent=2),flush=True)


def verify_frozen():
    check_protocol()
    frozen=json.loads((HERE/"frozen/selection.json").read_text())
    manifest=json.loads((HERE/"frozen/configuration_manifest.json").read_text())
    assert sha(HERE/"frozen/selection.json")==manifest["selection_sha256"]
    assert sha(ROOT/frozen["feature_definitions_path"])==frozen["feature_definitions_sha256"]
    access=frozen["semantic_data_access"]
    assert sha(ROOT/access["source"])==access["source_sha256"]
    for name,expected in frozen["source_hashes"].items():assert sha(HERE/name)==expected,name
    for name,expected in frozen["historical_source_hashes"].items():assert sha(ROOT/name)==expected,name
    for name,expected in frozen["artifacts_sha256"].items():assert sha(HERE/name)==expected,name
    for item in frozen["configurations"]:
        assert digest(item["features"])==item["feature_list_sha256"]
        assert item["features"]==[f for f in FEATURES if f in item["features"]]
        assert item["missing_indicators"]==[f"missing::{f}" for f in item["features"]]
        assert item["effective_columns"]==CALENDAR+item["features"]+item["missing_indicators"]
    assert frozen["future_evaluation"]["executed"] is False
    print("Frozen configuration valid; final evaluation not executed",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["create","verify"])
    action=parser.parse_args().action
    if action=="create":freeze()
    else:verify_frozen()
