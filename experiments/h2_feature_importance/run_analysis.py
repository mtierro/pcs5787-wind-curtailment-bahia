"""Reproduce historical predictions and interpret frozen models; never fit.

Run with repository .conda-env/bin/python -B. Only this experiment directory is
written. Historical objects are loaded, transformed and predicted without fitting.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

from provenance import HERE, ROOT, sha, json_write, verify
from core import (CALENDAR, FAMILIES, FAMILY_LABELS, block_map, perturb, scores,
                  importance_parts, local_examples, strata)

sys.path.insert(0, str(ROOT / "src"))
CONFIG = json.loads((HERE / "protocol.json").read_text())
GROUPS = json.loads((ROOT / "configs/v2/feature_groups.json").read_text())
FEATURES = sum((GROUPS[g] for g in ["basic", "engineered", "other_weather", "spatial"]), [])
assert len(FEATURES) == 51 and len(set(FEATURES)) == 51
FINAL_IDS = [
    ("54ca092bcc01b02c2097", "2ebe49d5e5d6e7cdbc2b"),
    ("d9ecad6f5877a9366330", "a64f59db40ea883738f8"),
    ("9b40f8cb39d3ea3770d2", "6b56746b5ad9e01c33a6"),
    ("22c5e9a7b13440b2c722", "0fe77d369254ea9cc4fa"),
    ("2ca2134204263d25fc5b", "28b4f9c60e5fc644f75a"),
    ("3c5196e98b4d2f39495b", "dcc0d53e6ac8e1fbcdda")]
DEV_IDS = [("591ad7d57961cce0b0df", "00fc1de073369b535078"),
           ("ef28660940fc1722d193", "a8244fbf71823588a643"),
           ("264af0a1c9d2e425b94d", "188914291adf201d7581")]
DEV_BINARIES = [("429a917f22af9ece41e8a8e7", "9e2cfd07891daa2362bfddce"),
                ("87eb6898c07cbb3919d1a5e4", "b5b6e6b261fbab1147b89452"),
                ("4cce25520ea949f1d5c33af7", "85d9cff6f5dc02c87a14336a")]


def output(split):
    p = HERE / split
    p.mkdir(exist_ok=True)
    return p


def load_base():
    columns = ["time_local", "y_ENE_02"] + CALENDAR + FEATURES
    base = pd.read_parquet(ROOT / "data/processed/v2/analytical.parquet", columns=columns)
    assert not base.duplicated(["time_local", "region"]).any()
    assert not np.isinf(base[FEATURES].to_numpy()).any()
    return base.sort_values(["time_local", "region"]).reset_index(drop=True)


def load_context(split):
    baseline = json.loads((HERE / "provenance/baseline.json").read_text())
    assert baseline["protocol_sha256"] == sha(HERE / "protocol.json")
    base = load_base()
    filters = [("reason", "==", "ENE"), ("population", "==", "spatial_recent"),
               ("seed", "==", 5787), ("variant", "in", ["A", "B"])]
    if split == "evaluation":
        filters += [("protocol", "==", "P2")]
        ref = pd.read_parquet(ROOT / "data/processed/v2/final_predictions.parquet", filters=filters)
        assert ref.calibration.eq("none").all()
        assert np.array_equal(ref.p, ref.p_raw)
        ids = FINAL_IDS
    else:
        filters += [("panel", "==", "quad")]
        ref = pd.read_parquet(ROOT / "data/processed/v2/selected_oof.parquet", filters=filters)
        ids = DEV_IDS
    assert set(ref.run_id) == {r for pair in ids for r in pair}
    contexts, checks, parameters, baseline_rows = [], [], [], []
    offset = 0
    for k, pair in enumerate(ids):
        frames = [ref.loc[ref.run_id.eq(r)].sort_values(["time_local", "region"]).reset_index(drop=True) for r in pair]
        pd.testing.assert_frame_equal(frames[0][["time_local", "region", "y"]],
                                      frames[1][["time_local", "region", "y"]], check_dtype=False)
        assert not frames[1].duplicated(["time_local", "region"]).any()
        meta_b = json.loads((ROOT / f".scratch/v2/cache/{pair[1]}.json").read_text())
        origin = meta_b["fold"]["name"]
        start, stop = pd.Timestamp(meta_b["fold"]["start"]), pd.Timestamp(meta_b["fold"]["stop"])
        full = base.loc[base.time_local.ge(start) & base.time_local.lt(stop)].reset_index(drop=True)
        times = pd.date_range(start.tz_convert(full.time_local.dt.tz), stop.tz_convert(full.time_local.dt.tz),
                              freq="h", inclusive="left").as_unit("us")
        regions = np.sort(full.region.unique())
        expected_keys = pd.MultiIndex.from_product([times, regions], names=["time_local", "region"])
        actual_keys = pd.MultiIndex.from_frame(full[["time_local", "region"]])
        assert len(regions) == 8 and actual_keys.equals(expected_keys)
        positions = actual_keys.get_indexer(pd.MultiIndex.from_frame(frames[1][["time_local", "region"]]))
        assert (positions >= 0).all()
        eligible = full.y_ENE_02.notna() & full.sp_wind.notna()
        assert np.array_equal(positions, np.flatnonzero(eligible))
        assert np.array_equal(full.y_ENE_02.to_numpy()[positions], frames[1].y.to_numpy())
        for variant_idx, variant in enumerate(["A", "B"]):
            run_id = pair[variant_idx]
            meta = json.loads((ROOT / f".scratch/v2/cache/{run_id}.json").read_text())
            if split == "evaluation":
                model_path = ROOT / f"models/v2/{run_id}.joblib"
                binary_meta = meta
            else:
                model_path = ROOT / f"models/v3/base_{DEV_BINARIES[k][variant_idx]}.joblib"
                binary_meta = json.loads(model_path.with_suffix(".json").read_text())
                for key in ["params", "strategy", "seed", "calendar", "variant", "population"]:
                    assert binary_meta["identity"]["spec"][key] == meta["spec"][key]
                assert not (ROOT / f"models/v2/{run_id}.joblib").exists()
            assert sha(model_path) == binary_meta["model_sha256"]
            pipe = joblib.load(model_path)
            prep = pipe.named_steps["representation"]
            clf = pipe.named_steps["model"]
            booster = clf.booster_
            assert list(clf.classes_) == [0, 1]
            assert clf.objective == "binary" and booster.params.get("sigmoid", 1) == 1
            assert list(prep.cat_) == CALENDAR and prep.tree and not prep.anomaly
            assert list(prep.numeric) == (FEATURES if variant == "B" else []) == meta["columns"]
            assert booster.params["categorical_column"] == list(range(6))
            expected = dict(num_leaves=7, min_child_samples=200, n_estimators=180,
                            learning_rate=.05, reg_lambda=10 if variant == "B" else 40,
                            random_state=5787, deterministic=True, force_col_wise=True)
            for param, value in expected.items():
                assert clf.get_params()[param] == value, (param, clf.get_params()[param])
            assert booster.num_trees() == 180
            zfull = prep.transform(full)
            z = zfull[positions]
            assert z.shape[1] == (108 if variant == "B" else 6)
            if variant == "B":
                assert np.array_equal(zfull[:, 6:57], full[FEATURES].to_numpy(), equal_nan=True)
                assert np.array_equal(zfull[:, 57:], (~np.isfinite(zfull[:, 6:57])).astype(float))
            p = booster.predict(z, num_threads=CONFIG["threads"])
            raw = booster.predict(z, raw_score=True, num_threads=CONFIG["threads"])
            ppipe = pipe.predict_proba(full.iloc[positions])[:, 1]
            refp = frames[variant_idx].p.to_numpy()
            residual = float(np.max(np.abs(p - refp)))
            assert residual <= CONFIG["probability_atol"]
            assert np.max(np.abs(expit(raw) - refp)) <= CONFIG["probability_atol"]
            assert np.array_equal(p, ppipe)
            ap, brier = scores(frames[variant_idx].y, p)
            checks.append(dict(split=split, origin=origin, variant=variant, run_id=run_id,
                               model_path=str(model_path.relative_to(ROOT)), model_sha256=sha(model_path),
                               n=len(p), positive=int(frames[variant_idx].y.sum()), columns=z.shape[1],
                               probability_max_residual=residual,
                               sigmoid_max_residual=float(np.max(np.abs(expit(raw) - refp))),
                               month_unknown=int(np.isnan(z[:, 3]).sum()),
                               region_month_unknown=int(np.isnan(z[:, 5]).sum()),
                               ap=ap, brier=brier, train_start=meta["train_start"], train_end=meta["train_end"],
                               n_train=meta["n_train"], original_binary=split == "evaluation"))
            parameters.append(dict(origin=origin, variant=variant, source=str(model_path.relative_to(ROOT)),
                                   historical_metadata=meta, binary_metadata=binary_meta,
                                   estimator=clf.get_params(), booster=booster.params,
                                   encoder_categories=[x.tolist() for x in prep.encoder_.categories_]))
            baseline_rows.append(frames[variant_idx].assign(origin=origin))
            if variant == "B":
                obs = frames[1][["time_local", "region", "y", "p", "threshold", "run_id"]].copy()
                obs["origin"] = origin
                obs["row_id"] = np.arange(offset, offset + len(obs))
                contexts.append(dict(origin=origin, obs=obs, full=full, times=times, regions=regions,
                                     positions=positions, z=z, zfull=zfull, booster=booster,
                                     raw=raw, offset=offset))
                offset += len(obs)
    obs = pd.concat([c["obs"] for c in contexts], ignore_index=True)
    assert len(obs) == (34811 if split == "evaluation" else 85185)
    ref_all = pd.concat(baseline_rows, ignore_index=True)
    baselines = []
    for variant, df in ref_all.groupby("variant"):
        for scope, group, pos in strata(df.reset_index(drop=True)):
            sub = df.iloc[pos]
            ap, brier = scores(sub.y, sub.p)
            pred = sub.p.ge(sub.threshold)
            tp = int((pred & sub.y.eq(1)).sum()); fp = int((pred & sub.y.eq(0)).sum())
            fn = int((~pred & sub.y.eq(1)).sum()); tn = int((~pred & sub.y.eq(0)).sum())
            baselines.append(dict(variant=variant, scope=scope, group=group, n=len(sub), ap=ap,
                                  brier=brier, tp=tp, fp=fp, fn=fn, tn=tn,
                                  precision=tp / (tp + fp) if tp + fp else np.nan,
                                  recall=tp / (tp + fn) if tp + fn else np.nan,
                                  f1=2 * tp / (2 * tp + fp + fn)))
    return contexts, obs, pd.DataFrame(checks), parameters, pd.DataFrame(baselines)


def unit(name):
    if "coverage" in name or name in ["gust_ratio", "direction_sin_lag2", "direction_cos_lag2"]: return "adimensional"
    if name == "sp_count": return "estações"
    if name == "sp_distance": return "km"
    if "temperature" in name or "temperatura" in name: return "°C"
    if "umidade" in name: return "%"
    if "pressure" in name: return "hPa"
    if "radiacao" in name: return "kJ/m²"
    if "precipitacao" in name: return "mm"
    if name == "wind_cubic_proxy": return "(m/s)³"
    return "m/s"


def inventory():
    rows = []
    for i, name in enumerate(FEATURES):
        family = next(g for g, idx in FAMILIES.items() if i in idx)
        operational = next(g for g in ["basic", "engineered", "other_weather", "spatial"] if name in GROUPS[g])
        rows.append(dict(index=i, feature=name, family=family, family_label=FAMILY_LABELS[family],
                         operational_group=operational, unit=unit(name), value_column=6 + i,
                         flag_column=57 + i, flag_alias=f"missing::{name}",
                         missing_policy="NaN nativo + flag(~isfinite); sem imputação no ramo tree",
                         latest_admissible="t-2h ou anterior; nomes lag históricos não são atraso físico"))
    pd.DataFrame(rows).to_csv(HERE / "feature_inventory.csv", index=False)
    effective = [dict(column=i, name=c, component="calendar", logical_variable=c) for i, c in enumerate(CALENDAR)]
    for component, start in [("value", 6), ("flag", 57)]:
        effective += [dict(column=start+i, name=c if component == "value" else f"missing::{c}",
                           component=component, logical_variable=c) for i, c in enumerate(FEATURES)]
    pd.DataFrame(effective).to_csv(HERE / "effective_columns.csv", index=False)


def preflight():
    verify()
    inventory()
    for split in CONFIG["splits"]:
        ctx, obs, checks, params, baseline = load_context(split)
        out = output(split)
        obs.to_parquet(out / "observations.parquet", index=False)
        checks.to_csv(out / "historical_verification.csv", index=False)
        baseline.to_csv(out / "historical_metrics.csv", index=False)
        json_write(out / "model_parameters.json", params)
        if split == "evaluation":
            local_examples(obs).to_csv(out / "local_example_selection.csv", index=False)
        print(f"Preflight {split}: {len(obs)} paired rows; max residual {checks.probability_max_residual.max()}", flush=True)


def shap_tables(obs, phi, values):
    variable_rows, family_rows, calendar_rows = [], [], []
    for scope, group, pos in strata(obs, interaction=True):
        pv, pf = phi[pos, 6:57], phi[pos, 57:108]
        stats = importance_parts(pv, pf)
        missing = ~np.isfinite(values[pos])
        for j, name in enumerate(FEATURES):
            joint = pv[:, j] + pf[:, j]
            variable_rows.append(dict(scope=scope, group=group, n=len(pos), feature=name,
                family=next(f for f, idx in FAMILIES.items() if j in idx),
                **{k: float(v[j]) for k, v in stats.items()}, missing_fraction=float(missing[:, j].mean()),
                mean_abs_joint_present=float(np.abs(joint[~missing[:, j]]).mean()) if (~missing[:, j]).any() else np.nan,
                mean_abs_joint_missing=float(np.abs(joint[missing[:, j]]).mean()) if missing[:, j].any() else np.nan))
        sums = []
        for family, indices in {**FAMILIES, "all_meteorology": list(range(51))}.items():
            v, f = pv[:, indices].sum(axis=1), pf[:, indices].sum(axis=1)
            s = importance_parts(v, f)
            family_rows.append(dict(scope=scope, group=group, n=len(pos), family=family,
                                    **{k: float(value) for k, value in s.items()},
                                    sum_logical_mean_abs=float(np.abs(pv[:, indices] + pf[:, indices]).mean(axis=0).sum()),
                                    sum_entry_mean_abs=float(np.abs(pv[:, indices]).mean(axis=0).sum() + np.abs(pf[:, indices]).mean(axis=0).sum())))
            if family != "all_meteorology": sums.append(v + f)
        assert np.max(np.abs(np.sum(sums, axis=0) - (pv + pf).sum(axis=1))) < CONFIG["decomposition_atol"]
        for j, name in enumerate(CALENDAR):
            calendar_rows.append(dict(scope=scope, group=group, n=len(pos), feature=name,
                                      mean_abs=float(np.abs(phi[pos, j]).mean()), mean_signed=float(phi[pos, j].mean())))
        calendar_rows.append(dict(scope=scope, group=group, n=len(pos), feature="all_calendar",
                                  mean_abs=float(np.abs(phi[pos, :6].sum(axis=1)).mean()),
                                  mean_signed=float(phi[pos, :6].sum(axis=1).mean())))
    return pd.DataFrame(variable_rows), pd.DataFrame(family_rows), pd.DataFrame(calendar_rows)


def shap_stage(split):
    contexts, obs, checks, _, _ = load_context(split)
    out = output(split)
    pd.testing.assert_frame_equal(obs, pd.read_parquet(out / "observations.parquet"))
    chunks, vals, qc = [], [], []
    for c in contexts:
        phi = c["booster"].predict(c["z"], pred_contrib=True, num_threads=CONFIG["threads"])
        assert phi.shape == (len(c["obs"]), 109)
        assert np.isfinite(phi).all()
        residual = float(np.max(np.abs(phi.sum(axis=1) - c["raw"])))
        assert residual <= CONFIG["margin_atol"]
        prob_error = float(np.max(np.abs(expit(phi.sum(axis=1)) - c["obs"].p.to_numpy())))
        assert prob_error <= CONFIG["probability_atol"]
        repeat = c["booster"].predict(c["z"][:256], pred_contrib=True, num_threads=CONFIG["threads"])
        assert np.array_equal(repeat, phi[:256])
        assert np.ptp(phi[:, -1]) == 0
        qc.append(dict(origin=c["origin"], n=len(phi), base_value=phi[0, -1],
                       margin_max_residual=residual, probability_max_residual=prob_error,
                       repeat_max_residual=float(np.max(np.abs(repeat - phi[:256]))), positive_class=1,
                       scale="log-odds", contribution_columns=108))
        chunks.append(phi); vals.append(c["z"][:, 6:57])
        print(f"SHAP {split} {c['origin']}: additive error={residual:.3g}", flush=True)
    phi, values = np.concatenate(chunks), np.concatenate(vals)
    np.savez_compressed(out / "shap_contributions.npz", phi=phi)
    pd.DataFrame(qc).to_csv(out / "shap_verification.csv", index=False)
    variable, family, calendar = shap_tables(obs, phi, values)
    variable.to_csv(out / "shap_variables.csv", index=False)
    family.to_csv(out / "shap_families.csv", index=False)
    calendar.to_csv(out / "shap_calendar.csv", index=False)
    stability = []
    for scope in ["origin", "region"]:
        table = variable.loc[variable.scope.eq(scope)].pivot(index="feature", columns="group", values="mean_abs_joint")
        for left in table:
            for right in table:
                a = set(table[left].sort_values(ascending=False, kind="stable").head(10).index)
                b = set(table[right].sort_values(ascending=False, kind="stable").head(10).index)
                stability.append(dict(scope=scope, left=left, right=right,
                                      spearman=float(spearmanr(table[left], table[right]).statistic),
                                      top10_jaccard=len(a & b) / len(a | b)))
    pd.DataFrame(stability).to_csv(out / "shap_stability.csv", index=False)
    if split == "evaluation":
        selected = pd.read_csv(out / "local_example_selection.csv")
        local_rows = []
        for row in selected.itertuples():
            ix = int(row.row_id)
            joint = phi[ix, 6:57] + phi[ix, 57:108]
            for j, name in enumerate(CALENDAR + FEATURES):
                contribution = phi[ix, j] if j < 6 else joint[j-6]
                local_rows.append(dict(row_id=ix, origin=row.origin, cell=row.cell, feature=name,
                                       contribution=contribution, value_contribution=phi[ix, j] if j < 6 else phi[ix, j],
                                       flag_contribution=0 if j < 6 else phi[ix, 57+j-6],
                                       value=str(obs.iloc[ix].region) if j == 0 else (values[ix, j-6] if j >= 6 else np.nan),
                                       base_value=phi[ix, -1], raw_margin=phi[ix].sum(), p=obs.iloc[ix].p))
        pd.DataFrame(local_rows).to_csv(out / "local_contributions.csv", index=False)


def permutation_stage(split):
    contexts, obs, _, _, _ = load_context(split)
    out = output(split)
    pd.testing.assert_frame_equal(obs, pd.read_parquet(out / "observations.parquet"))
    y, original = obs.y.to_numpy(), obs.p.to_numpy()
    masks = list(strata(obs))
    baseline = {(scope, group): scores(y[idx], original[idx]) for scope, group, idx in masks}
    targets = [(name, "variable", [i]) for i, name in enumerate(FEATURES)]
    targets += [(name, "family", indices) for name, indices in FAMILIES.items()]
    targets += [("all_meteorology", "all", list(range(51)))]
    map_rows = []
    for c in contexts:
        c["maps"] = {}
        for hours in CONFIG["block_hours"]:
            for rep in range(CONFIG["repetitions"]):
                parts = (CONFIG["permutation_seed"], split, c["origin"], rep)
                mapping, diagnostics = block_map(c["times"], hours, parts)
                repeated, _ = block_map(c["times"], hours, parts)
                assert np.array_equal(mapping, repeated)
                nr = len(c["regions"])
                # Keep destination region; all regions share identical time donors.
                fullmap = (mapping[:, None] * nr + np.arange(nr)).ravel()
                donor_positions = fullmap[c["positions"]]
                assert np.array_equal(c["full"].region.to_numpy()[donor_positions], c["obs"].region)
                assert np.array_equal(c["full"].time_local.dt.strftime("%Y-%m").to_numpy()[donor_positions],
                                      c["obs"].time_local.dt.strftime("%Y-%m"))
                c["maps"][(hours, rep)] = donor_positions
                import hashlib
                map_hash = hashlib.sha256(donor_positions.astype("<i8").tobytes()).hexdigest()
                for diagnostic in diagnostics:
                    map_rows.append(dict(origin=c["origin"], rep=rep, **diagnostic, map_sha256=map_hash,
                                         eligible_n=len(donor_positions),
                                         eligible_moved_fraction=float((donor_positions != c["positions"]).mean()),
                                         donor_unknown_y=int(c["full"].y_ENE_02.isna().to_numpy()[donor_positions].sum()),
                                         donor_missing_sp_wind=int(c["full"].sp_wind.isna().to_numpy()[donor_positions].sum())))
    pd.DataFrame(map_rows).to_csv(out / "permutation_maps.csv", index=False)
    started = time.monotonic()
    all_results = []
    for hours in CONFIG["block_hours"]:
        for target_index, (target, kind, indices) in enumerate(targets):
            for rep in range(CONFIG["repetitions"]):
                predicted = np.empty(len(obs))
                for c in contexts:
                    donor = c["zfull"][c["maps"][(hours, rep)]]
                    z = perturb(c["z"], donor, indices)
                    p = c["booster"].predict(z, num_threads=CONFIG["threads"])
                    assert p.shape == (len(c["obs"]),) and np.isfinite(p).all()
                    if kind == "all" and rep == 0:
                        repeat = c["booster"].predict(perturb(c["z"], donor, indices), num_threads=CONFIG["threads"])
                        assert np.array_equal(p, repeat)
                    predicted[c["offset"]:c["offset"] + len(p)] = p
                for scope, group, idx in masks:
                    ap, brier = scores(y[idx], predicted[idx])
                    ap0, b0 = baseline[(scope, group)]
                    all_results.append(dict(target=target, kind=kind, hours=hours, rep=rep,
                                            scope=scope, group=group, n=len(idx), ap_original=ap0,
                                            brier_original=b0, ap_permuted=ap, brier_permuted=brier,
                                            i_ap=ap0-ap, i_brier=brier-b0))
            print(f"PI {split} {hours}h {target_index+1}/56 {target}; elapsed={time.monotonic()-started:.1f}s", flush=True)
        # Checkpoint complete block-length only, no individual permutation predictions.
        pd.DataFrame(all_results).to_parquet(out / "permutation_replicates.parquet", index=False)
    df = pd.DataFrame(all_results)
    keys = ["target", "kind", "hours", "scope", "group", "n"]
    summaries = []
    for key, frame in df.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        assert len(frame) == CONFIG["repetitions"]
        row["repetitions"] = len(frame)
        for metric in ["i_ap", "i_brier"]:
            values = frame[metric]
            for suffix, value in [("mean", values.mean()), ("std", values.std()),
                                  ("p05", values.quantile(.05)), ("p95", values.quantile(.95)),
                                  ("min", values.min()), ("max", values.max())]:
                row[f"{metric}_{suffix}"] = value
            row[f"{metric}_negative_repetitions"] = int(values.lt(0).sum())
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "permutation_summary.csv", index=False)
    sv = pd.read_csv(out / "shap_variables.csv")
    merged = sv.loc[sv.scope.eq("global"), ["feature", "mean_abs_joint", "mean_abs_value", "mean_abs_flag"]]
    for hours in CONFIG["block_hours"]:
        p = summary.loc[summary.scope.eq("global") & summary.kind.eq("variable") & summary.hours.eq(hours)]
        merged = merged.merge(p[["target", "i_ap_mean", "i_brier_mean"]].rename(columns={
            "target": "feature", "i_ap_mean": f"i_ap_{hours}", "i_brier_mean": f"i_brier_{hours}"}), on="feature", validate="one_to_one")
    merged.to_csv(out / "ranking_comparison.csv", index=False)
    correlations = []
    for metric in ["i_ap_24", "i_ap_168", "i_brier_24", "i_brier_168"]:
        correlations.append(dict(metric=metric, spearman=float(spearmanr(merged.mean_abs_joint, merged[metric]).statistic)))
    pd.DataFrame(correlations).to_csv(out / "ranking_correlations.csv", index=False)
    json_write(out / "permutation_execution.json", dict(seconds=time.monotonic()-started,
               predictions_not_retained=True, rows=len(df), target_count=len(targets),
               max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               population_preserved=True, calendar_preserved=True, pairs_preserved=True,
               seed_maps_repeated_exactly=True, all_weather_first_rep_predictions_repeated_exactly=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preflight", "shap", "permutation"])
    parser.add_argument("--split", choices=CONFIG["splits"], default="evaluation")
    args = parser.parse_args()
    start = time.monotonic()
    if args.stage == "preflight": preflight()
    elif args.stage == "shap": shap_stage(args.split)
    else: permutation_stage(args.split)
    json_write(HERE / f"provenance/execution_{args.stage}_{args.split}.json", dict(
        utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-start,
        max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        command=sys.argv, protocol_sha256=sha(HERE / "protocol.json"),
        code_sha256={name: sha(HERE / name) for name in ["run_analysis.py", "core.py", "provenance.py"]}))


if __name__ == "__main__":
    main()
