"""Temporal, population, selection and bootstrap invariants, before training."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score
from selection_core import (FEATURES, PAIRS, CALENDAR, ORIGINS, FREEZE, DEV_STOP, bounds, cutoff,
                            top_k, columns_for, threshold_from_history, ValueOnlyRepresentation,
                            Representation, WeightedAP, VARIANTS, strictly_align)


def test_clock_outer_inner_and_truncated_march():
    assert len(ORIGINS)==15
    assert str(DEV_STOP)=="2026-03-24 23:00:00-03:00"
    for origin in ORIGINS+[FREEZE]:
        b=bounds(origin)
        last=b["train_stop"]-pd.Timedelta(hours=1)
        assert last==origin-pd.Timedelta(hours=170)
        assert last+pd.Timedelta(hours=169)==origin-pd.Timedelta(hours=1)
        assert b["train_start"]<b["inner_stop"]<b["rank_origin"]<b["rank_stop"]<origin
        assert b["inner_stop"]-pd.Timedelta(hours=1)+pd.Timedelta(hours=169)==b["rank_origin"]-pd.Timedelta(hours=1)
        assert b["validation_stop"]<=DEV_STOP
    assert bounds(ORIGINS[-1])["validation_stop"]==DEV_STOP


@pytest.mark.parametrize("k",[5,10,20])
def test_top_k_determinism_pair_closure_and_historical_order(k):
    scores=np.zeros(51);scores[FEATURES.index("direction_cos_lag2")]=100
    scores[FEATURES.index("sp_v")]=90
    cols=top_k(scores,k)
    assert cols==top_k(scores.copy(),k)
    assert k<=len(cols)<=k+1
    assert cols==[c for c in FEATURES if c in cols]
    for pair in PAIRS: assert set(cols).isdisjoint(pair) or set(pair)<=set(cols)
    assert "direction_sin_lag2" in cols
    # Zero importance ties use historical order, not alphabetical order.
    assert top_k(np.zeros(51),5)==FEATURES[:5]


def test_structural_sizes_and_no_flag_control():
    scores=np.arange(51,dtype=float)
    expected={"A":0,"B":51,"C_spatial":14,"B_minus_anchor":43,"B_minus_temporal":31,
              "B_minus_atmospheric":42,"B_minus_spatial":37,"B_no_indicators":51}
    for variant,k in expected.items(): assert len(columns_for(variant,scores))==k
    frame=pd.DataFrame({"region":["a","b","a"],"hour":[1,2,3],"weekday":[1,1,1],"month":[0,0,0],
                        "region_hour":["a_1","b_2","a_3"],"region_month":["a_0","b_0","a_0"],
                        FEATURES[0]:[np.nan,2,3]})
    original=Representation((FEATURES[0],),"strong",True).fit(frame)
    control=ValueOnlyRepresentation((FEATURES[0],),"strong",True).fit(frame)
    valid=frame.copy();valid["month"]=1;valid["region_month"]=["a_1","b_1","a_1"]
    a,b=original.transform(valid),control.transform(valid)
    assert a.shape==(3,8) and b.shape==(3,7)
    assert np.array_equal(a[:,:7],b,equal_nan=True)
    assert np.isnan(b[:,3]).all() and np.isnan(b[:,5]).all()
    assert np.array_equal(a[:,7],~np.isfinite(a[:,6]))


def test_threshold_excludes_unavailable_and_current_validation_labels():
    origin=ORIGINS[1];end=cutoff(origin)
    times=pd.date_range(end-pd.Timedelta(hours=250),periods=250,freq="h")
    h=pd.DataFrame({"time_local":times,"y":np.tile([0,1],125),"p":np.tile([.1,.9],125)})
    t0,record=threshold_from_history(h,origin)
    future=pd.DataFrame({"time_local":[end,origin,origin+pd.Timedelta(hours=1)],"y":[1,1,0],"p":[.001,.002,.999]})
    t1,rec1=threshold_from_history(pd.concat([h,future]),origin)
    assert t1==t0 and rec1==record and record["n"]==250
    assert threshold_from_history(None,ORIGINS[0])[0]==.5


def test_weighted_ap_matches_sklearn_with_ties_and_zero_weight_rows():
    y=np.array([0,1,1,0,1,0]);p=np.array([.4,.4,.8,.1,.8,.9]);w=np.array([2,1,0,3,2,0])
    assert abs(WeightedAP(y,p)(w)-average_precision_score(y,p,sample_weight=w))<1e-14


def test_population_alignment_rejects_silent_row_loss_or_label_change():
    frames=[]
    for variant in VARIANTS:
        frames.append(pd.DataFrame(dict(row_id=[1,2],time_local=pd.date_range("2025-01-01",periods=2,freq="h"),
                          region=["a","b"],y=[0,1],origin=["2025-01"]*2,p=[.1,.9],threshold=[.5,.5],variant=variant)))
    pred=pd.concat(frames,ignore_index=True)
    assert len(strictly_align(pred))==11
    with pytest.raises(AssertionError):strictly_align(pred.drop(index=0))
    changed=pred.copy();changed.loc[0,"y"]=1
    with pytest.raises(AssertionError):strictly_align(changed)


def test_feature_clock_original_functions_ignore_t_minus_one_and_future():
    from pcs5787.features import historical_features
    from pcs5787.v2.data import lagged_weather
    grid=pd.date_range("2024-10-01",periods=200,freq="h",tz="America/Bahia")
    names=["vento_ms","rajada_ms","temperatura_c","umidade_pct","pressure_hpa","radiacao_kjm2",
           "precipitacao_mm","direction_sin","direction_cos","wind_u","wind_v"]
    raw=pd.DataFrame({name:np.arange(200,dtype=float)+i for i,name in enumerate(names)},index=grid)
    modified=raw.copy();modified.iloc[99:]=10000
    a=historical_features(raw,delay_hours=2);b=historical_features(modified,delay_hours=2)
    pd.testing.assert_frame_equal(a.iloc[:101],b.iloc[:101])
    a=lagged_weather(raw,grid);b=lagged_weather(modified,grid)
    pd.testing.assert_frame_equal(a.iloc[:101],b.iloc[:101])
