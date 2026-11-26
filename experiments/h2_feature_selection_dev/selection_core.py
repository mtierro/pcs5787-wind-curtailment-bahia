"""Temporal and selection primitives; original representation and estimator."""
import json
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from audit import HERE, ROOT, digest

sys.path.insert(0, str(ROOT/"src"))
from pcs5787.v2.learners import Representation, make_model
from pcs5787.v2.calibration import decision_threshold
from pcs5787.v2.evaluation import WeightedAP

CONFIG = json.loads((HERE/"protocol.json").read_text())
GROUPS = json.loads((ROOT/"configs/v2/feature_groups.json").read_text())
FEATURES = sum([GROUPS[g] for g in ["basic", "engineered", "other_weather", "spatial"]], [])
CALENDAR = ["region", "hour", "weekday", "month", "region_hour", "region_month"]
FAMILIES = {name:FEATURES[a:b] for name,(a,b) in CONFIG["families"].items()}
PAIRS = CONFIG["inner_selection"]["atomic_pairs"]
VARIANTS = CONFIG["variants"]
REDUCED = CONFIG["choice"]["eligible_candidates"]
assert len(FEATURES) == 51 and len(set(FEATURES)) == 51


def stamp(value):
    t = pd.Timestamp(value)
    return t.tz_localize(CONFIG["timezone"]) if t.tz is None else t.tz_convert(CONFIG["timezone"])


def cutoff(origin):
    return stamp(origin) - pd.Timedelta(hours=169)


FREEZE = stamp(CONFIG["freeze_origin"])
DEV_STOP = cutoff(FREEZE)
FIRST = stamp(CONFIG["first_origin"])
ORIGINS = list(pd.date_range(FIRST, stamp(CONFIG["last_origin"]), freq="MS"))
FIRST_TRAIN = cutoff(FIRST)-pd.DateOffset(months=3)


def bounds(origin):
    o = stamp(origin)
    assert FIRST <= o <= FREEZE
    stop = cutoff(o)
    rank_origin = o-pd.DateOffset(months=1)
    return dict(origin=o, train_start=stop-pd.DateOffset(months=3), train_stop=stop,
                validation_start=o, validation_stop=min(o+pd.DateOffset(months=1), DEV_STOP),
                rank_origin=rank_origin, inner_stop=cutoff(rank_origin), rank_stop=stop)


def assert_admissible(frame, origin):
    assert len(frame) and frame.time_local.max() < cutoff(origin)
    assert frame.time_local.max()+pd.Timedelta(hours=169) <= stamp(origin)-pd.Timedelta(hours=1)
    assert frame.time_local.min() >= FIRST_TRAIN and frame.time_local.max() < DEV_STOP
    assert frame.y_ENE_02.notna().all() and frame.sp_wind.notna().all()


def ranking_order(scores):
    scores = np.asarray(scores, dtype=float)
    assert scores.shape == (51,) and np.isfinite(scores).all() and (scores >= 0).all()
    return sorted(range(51), key=lambda j:(-scores[j], j))


def top_k(scores, k):
    chosen = set()
    pairs = {name:set(pair) for pair in PAIRS for name in pair}
    for j in ranking_order(scores):
        name = FEATURES[j]
        if name in chosen: continue
        chosen.update(pairs.get(name, {name}))
        if len(chosen) >= k: break
    columns = [name for name in FEATURES if name in chosen]
    assert k <= len(columns) <= k+1
    for pair in PAIRS: assert set(pair).isdisjoint(chosen) or set(pair) <= chosen
    return columns


def columns_for(variant, scores):
    if variant == "A": return []
    if variant in ["B", "B_no_indicators"]: return FEATURES.copy()
    if variant == "C_spatial": return FAMILIES["spatial"].copy()
    if variant.startswith("C_top"): return top_k(scores, int(variant[5:]))
    excluded = FAMILIES[variant.removeprefix("B_minus_")]
    return [f for f in FEATURES if f not in excluded]


class ValueOnlyRepresentation(Representation):
    """Explicit control: same fitted encoder/native NaNs, omit only flags."""
    def transform(self, x):
        represented = super().transform(x)
        return represented[:, :len(self.cat_)+len(self.numeric)]


def model_spec(variant):
    return dict(reason="ENE", strategy="window3", population="spatial_recent", calendar="strong",
                seed=CONFIG["model"]["seed"], algorithm="lightgbm", variant="A" if variant == "A" else "B",
                params={**CONFIG["model"]["params"], "l2":40 if variant == "A" else 10})


def fit_model(frame, columns, variant, origin, role):
    assert role in ["inner", "outer", "repeat_inner", "repeat_outer"]
    assert stamp(origin) < FREEZE, "External fit at April or later is prohibited in this mission"
    assert_admissible(frame, origin)
    assert columns == [c for c in FEATURES if c in columns]
    model = make_model(model_spec(variant), columns)
    if variant == "B_no_indicators":
        model.set_params(representation=ValueOnlyRepresentation(**model.named_steps["representation"].get_params()))
    started = time.perf_counter()
    with threadpool_limits(limits=4), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(frame, frame.y_ENE_02.to_numpy(dtype=int), model__categorical_feature=list(range(6)))
    elapsed = time.perf_counter()-started
    prep, clf = model.named_steps["representation"], model.named_steps["model"]
    assert list(clf.classes_) == [0,1]
    assert clf.booster_.num_trees() == 180 and clf.objective == "binary"
    assert clf.booster_.params["categorical_column"] == list(range(6))
    assert prep.cat_ == CALENDAR and list(prep.numeric) == columns
    return model, elapsed, sorted(set(str(w.message) for w in caught))


def predict(model, frame, columns, flags=True):
    assert len(frame) and frame.time_local.max() < DEV_STOP
    started = time.perf_counter()
    prep = model.named_steps["representation"]
    z = prep.transform(frame)
    k = len(columns)
    assert z.shape == (len(frame), 6+k*(2 if flags else 1))
    assert np.array_equal(z[:, 6:6+k], frame[columns].to_numpy(float), equal_nan=True)
    if flags and k:
        assert np.array_equal(z[:, 6+k:], (~np.isfinite(z[:, 6:6+k])).astype(float))
    p = model.named_steps["model"].booster_.predict(z, num_threads=4)
    assert np.isfinite(p).all()
    return p, z, time.perf_counter()-started


def threshold_from_history(history, origin):
    end = cutoff(origin)
    if history is None or len(history) == 0:
        return .5, dict(n=0, first=None, last=None, fallback=True)
    past = history.loc[history.time_local.ge(end-pd.Timedelta(days=90)) & history.time_local.lt(end)]
    assert len(past) == 0 or past.time_local.max()+pd.Timedelta(hours=169) <= stamp(origin)-pd.Timedelta(hours=1)
    threshold = decision_threshold(past.y.to_numpy(), past.p.to_numpy())
    return threshold, dict(n=len(past), first=str(past.time_local.min()) if len(past) else None,
                           last=str(past.time_local.max()) if len(past) else None,
                           fallback=bool(len(past)<100 or past.y.sum()==0))


def score(y, p, threshold):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = p >= np.asarray(threshold)
    tp = int(((y==1)&pred).sum()); fp = int(((y==0)&pred).sum())
    fn = int(((y==1)&~pred).sum()); tn = int(((y==0)&~pred).sum())
    return dict(n=len(y), positive=int(y.sum()), prevalence=float(y.mean()),
                ap=float(average_precision_score(y,p)), brier=float(np.mean((p-y)**2)),
                roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                tp=tp, fp=fp, fn=fn, tn=tn, alert_region_hours=tp+fp,
                alert_fraction=float(pred.mean()), mean_p=float(p.mean()))


def frame_fingerprint(frame):
    return digest({"row_id":frame.row_id.tolist(), "y":frame.y_ENE_02.tolist()})


def strictly_align(predictions):
    reference = predictions.loc[predictions.variant.eq("B")].sort_values("row_id").reset_index(drop=True)
    assert not reference.duplicated(["region", "time_local"]).any()
    result = {}
    for variant in VARIANTS:
        frame = predictions.loc[predictions.variant.eq(variant)].sort_values("row_id").reset_index(drop=True)
        assert len(frame) == len(reference)
        for col in ["row_id", "time_local", "region", "y", "origin"]:
            assert np.array_equal(frame[col].to_numpy(), reference[col].to_numpy()), (variant, col)
        result[variant] = frame
    return result
