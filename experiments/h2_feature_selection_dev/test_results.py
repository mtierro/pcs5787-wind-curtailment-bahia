"""Independent checks of executed development artifacts; never train a model."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.metrics import average_precision_score
from audit import HERE, sha, digest, check_protocol
from selection_core import (FEATURES,CALENDAR,VARIANTS,REDUCED,ORIGINS,FREEZE,DEV_STOP,FIRST,
                            bounds,stamp,columns_for,strictly_align,threshold_from_history,fit_model)
from campaign import load_development,frame_fingerprint
from analyze import choose


def read_json(name):return json.loads((HERE/name).read_text())


@pytest.fixture(scope="module")
def data():
    if not (HERE/"predictions.parquet").exists():pytest.skip("Run development campaign first")
    base,access=load_development()
    pred=pd.read_parquet(HERE/"predictions.parquet")
    return base,pred,strictly_align(pred),access


def test_executed_populations_clock_and_exact_memberships(data):
    base,pred,aligned,access=data
    assert len(aligned)==11 and all(len(f)==85185 for f in aligned.values())
    assert access["final_period_rows_materialized"]==0
    assert not access["prior_importance_rankings_read"] and not access["final_models_loaded"]
    members=pd.read_parquet(HERE/"split_membership.parquet")
    for origin in ORIGINS:
        b=bounds(origin);label=origin.strftime("%Y-%m")
        expected={"outer_train":(b["train_start"],b["train_stop"]),
                  "outer_validation":(origin,b["validation_stop"]),
                  "inner_fit":(b["train_start"],b["inner_stop"]),
                  "inner_rank":(b["rank_origin"],b["rank_stop"])}
        ids={}
        for role,(start,stop) in expected.items():
            ids[role]=members.loc[members.origin.eq(label)&members.role.eq(role),"row_id"].to_numpy()
            correct=base.loc[base.time_local.ge(start)&base.time_local.lt(stop),"row_id"].to_numpy()
            assert np.array_equal(ids[role],correct)
        assert set(ids["inner_fit"]).isdisjoint(ids["inner_rank"])
        assert set(ids["inner_fit"])|set(ids["inner_rank"]) <= set(ids["outer_train"])
        assert set(ids["outer_train"]).isdisjoint(ids["outer_validation"])
        for frame in aligned.values():
            current=frame.loc[frame.origin.eq(label)]
            assert np.array_equal(current.row_id,ids["outer_validation"])
    assert pred.time_local.max()+pd.Timedelta(hours=169)==FREEZE-pd.Timedelta(hours=1)
    with pytest.raises(AssertionError,match="April"):
        fit_model(base,FEATURES,"B",FREEZE,"outer")


def test_saved_models_independently_reproduce_all_oof_probabilities(data):
    base,pred,aligned,access=data
    history={(r["variant"],r["origin"]):r for r in read_json("selection_history.json")}
    ranks=pd.read_csv(HERE/"inner_rankings.csv")
    for rec in read_json("model_ledger.json"):
        path=HERE/rec["model_path"];assert sha(path)==rec["model_sha256"]
        cols=history[(rec["variant"],rec["origin"])]["features"]
        scores=ranks.loc[ranks.origin.eq(rec["origin"])].sort_values("original_index").mean_abs_joint.to_numpy()
        assert cols==columns_for(rec["variant"],scores)
        saved=aligned[rec["variant"]].query("origin == @rec['origin']")
        frame=base.set_index("row_id",drop=False).loc[saved.row_id]
        model=joblib.load(path);prep=model.named_steps["representation"];clf=model.named_steps["model"]
        assert list(prep.numeric)==cols and prep.cat_==CALENDAR
        z=prep.transform(frame);k=len(cols)
        assert z.shape[1]==6+k*(1 if rec["variant"]=="B_no_indicators" else 2)
        assert np.array_equal(z[:,6:6+k],frame[cols].to_numpy(float),equal_nan=True)
        if rec["flags"]:assert np.array_equal(z[:,6+k:],~np.isfinite(z[:,6:6+k]))
        assert np.isnan(z[:,3]).all() and np.isnan(z[:,5]).all()
        assert list(clf.classes_)==[0,1]
        assert np.max(np.abs(clf.booster_.predict(z,num_threads=4)-saved.p.to_numpy()))<=1e-12
        assert np.array_equal(frame.y_ENE_02,saved.y)
        assert clf.reg_lambda==(40 if rec["variant"]=="A" else 10)
        assert clf.num_leaves==7 and clf.min_child_samples==200 and clf.n_estimators==180
        assert clf.learning_rate==.05 and clf.colsample_bytree==.9 and clf.subsample==.85
        assert clf.subsample_freq==1 and clf.random_state==5787
        assert clf.get_params()["deterministic"] and clf.get_params()["force_col_wise"]
    checks=pd.read_csv(HERE/"historical_anchor_checks.csv")
    assert len(checks)==6 and checks.probability_max_residual.eq(0).all()


def test_inner_shap_ranks_reconstructed_from_saved_models(data):
    base,_,_,_=data
    ranks=pd.read_csv(HERE/"inner_rankings.csv")
    for rec in read_json("inner_model_ledger.json"):
        assert sha(HERE/rec["model_path"])==rec["model_sha256"]
        frame=base.loc[base.time_local.ge(stamp(rec["rank_first"])) & base.time_local.le(stamp(rec["rank_last"]))]
        assert frame_fingerprint(frame)==rec["rank_hash"]
        model=joblib.load(HERE/rec["model_path"]);z=model.named_steps["representation"].transform(frame)
        booster=model.named_steps["model"].booster_
        phi=booster.predict(z,pred_contrib=True,num_threads=4)
        assert np.max(np.abs(phi.sum(axis=1)-booster.predict(z,raw_score=True,num_threads=4)))<1e-10
        assert np.max(np.abs(expit(phi.sum(axis=1))-booster.predict(z,num_threads=4)))<1e-12
        scores=np.abs(phi[:,6:57]+phi[:,57:108]).mean(axis=0)
        saved=ranks.loc[ranks.origin.eq(rec["origin"])].sort_values("original_index").mean_abs_joint.to_numpy()
        np.testing.assert_allclose(scores,saved,rtol=0,atol=1e-14)
    assert read_json("inner_model_ledger.json")[0]["repeat_shap_max_residual"]==0
    repeated=[r for r in read_json("model_ledger.json") if "repeat_probability_max_residual" in r]
    assert len(repeated)==4 and all(r["repeat_probability_max_residual"]==0 for r in repeated)


def test_thresholds_only_use_prior_available_oof(data):
    _,pred,aligned,_=data
    for rec in read_json("model_ledger.json"):
        origin=stamp(rec["origin"]+"-01");frame=aligned[rec["variant"]]
        earlier=frame.loc[frame.time_local.lt(origin)]
        threshold,record=threshold_from_history(earlier,origin)
        assert threshold==rec["threshold"] and record==rec["threshold_history"]


def test_bootstrap_seed_population_weights_and_rule(data):
    _,_,aligned,_=data
    reference=aligned["B"];y=reference.y.to_numpy(int)
    samples=pd.read_parquet(HERE/"bootstrap_replicates.parquet")
    assert samples.groupby(["variant","block_hours"]).size().eq(2000).all()
    assert samples.shape[0]==40000
    for diag in read_json("bootstrap_diagnostics.json"):
        nblocks=diag["blocks"];hours=diag["block_hours"]
        draws=np.random.default_rng(diag["seed"]).multinomial(nblocks,np.full(nblocks,1/nblocks),size=2000)
        assert hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()==diag["draw_sha256"]
        blocks=((reference.time_local-FIRST).dt.total_seconds().to_numpy()/3600//hours).astype(int)
        weights=draws[0,blocks]
        ap_b=average_precision_score(y,reference.p,sample_weight=weights)
        br_b=np.average((reference.p-y)**2,weights=weights)
        for variant in VARIANTS:
            if variant=="B":continue
            row=samples.loc[samples.variant.eq(variant)&samples.block_hours.eq(hours)&samples.rep.eq(0)].iloc[0]
            p=aligned[variant].p
            assert abs(average_precision_score(y,p,sample_weight=weights)-ap_b-row.delta_ap)<1e-12
            assert abs(np.average((p-y)**2,weights=weights)-br_b-row.delta_brier)<1e-12
    summary=pd.read_csv(HERE/"summary.csv");intervals=pd.read_csv(HERE/"paired_intervals.csv")
    qualification,decision=choose(summary,intervals)
    assert decision==read_json("selection_decision.json")
    for row in qualification.itertuples():
        tests=intervals.loc[intervals.variant.eq(row.variant)]
        assert row.qualified==bool((tests.ap_low>-.01).all() and (tests.ap_simultaneous_low>-.01).all())
    assert check_protocol()["protocol_sha256"]==read_json("provenance/campaign_execution.json")["protocol_sha256"]


def test_frozen_configuration_has_only_admissible_ranking(data):
    if not (HERE/"frozen/selection.json").exists():pytest.skip("Freeze after pre-freeze tests")
    from freeze import verify_frozen
    verify_frozen()
    base,_,_,_=data;frozen=read_json("frozen/selection.json")
    rec=frozen["selection_model"];b=bounds(FREEZE)
    assert stamp(rec["train_last"])<b["inner_stop"] and stamp(rec["rank_last"])<DEV_STOP
    assert rec["purpose"]=="freeze_ranking_only" and rec["rank_origin"]==str(b["rank_origin"])
    assert frozen["future_evaluation"]["executed"] is False
    assert read_json("provenance/freeze_execution.json")["external_april_fits"]==0
    ranked=pd.read_csv(HERE/"frozen/ranking_pre_april.csv").sort_values("original_index")
    for item in frozen["configurations"]:
        assert item["features"]==columns_for(item["variant"],ranked.mean_abs_joint.to_numpy())
        assert digest(item["features"])==item["feature_list_sha256"]
    assert len(list((HERE/"models").glob("*/*.joblib")))==165
