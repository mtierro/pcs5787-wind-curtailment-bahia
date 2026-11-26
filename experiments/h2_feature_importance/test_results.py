"""Integration tests against complete saved results, with historical references."""
import ast
import json
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from provenance import HERE, sha
from core import FAMILIES


@pytest.mark.parametrize("split,n,nmodels", [("evaluation", 34811, 6), ("development", 85185, 3)])
def test_predictions_contributions_and_aggregation(split, n, nmodels):
    path = HERE / split
    obs = pd.read_parquet(path / "observations.parquet")
    phi = np.load(path / "shap_contributions.npz")["phi"]
    qc = pd.read_csv(path / "historical_verification.csv")
    assert len(obs) == n and phi.shape == (n, 109)
    assert not obs.duplicated(["time_local", "region"]).any()
    assert len(qc) == 2*nmodels and qc.probability_max_residual.max() <= 1e-12
    assert (qc.columns == "columns").any()
    assert set(qc.loc[qc.variant.eq("B"), "columns"]) == {108}
    np.testing.assert_allclose(expit(phi.sum(axis=1)), obs.p, rtol=0, atol=1e-12)
    variable = pd.read_csv(path / "shap_variables.csv")
    for feature, frame in variable.loc[variable.scope.eq("origin")].groupby("feature"):
        global_value = variable.loc[variable.scope.eq("global") & variable.feature.eq(feature), "mean_abs_joint"].iloc[0]
        np.testing.assert_allclose(np.average(frame.mean_abs_joint, weights=frame.n), global_value, rtol=0, atol=1e-12)
    families = pd.read_csv(path / "shap_families.csv")
    joints = phi[:, 6:57]+phi[:, 57:108]
    for name, indices in {**FAMILIES, "all_meteorology": list(range(51))}.items():
        row = families.loc[families.scope.eq("global") & families.family.eq(name)].iloc[0]
        np.testing.assert_allclose(row.mean_abs_joint, np.abs(joints[:, indices].sum(axis=1)).mean(), atol=1e-12)
        np.testing.assert_allclose(row.sum_logical_mean_abs, np.abs(joints[:, indices]).mean(axis=0).sum(), atol=1e-12)


@pytest.mark.parametrize("split,origins", [("evaluation", 6), ("development", 3)])
def test_complete_permutations_sign_convention_and_population(split, origins):
    path = HERE / split
    reps = pd.read_parquet(path / "permutation_replicates.parquet")
    assert len(reps) == 56*2*20*(1+origins+8)
    keys = ["target", "hours", "scope", "group"]
    assert reps.groupby(keys).rep.nunique().eq(20).all()
    assert reps.groupby(keys).n.nunique().eq(1).all()
    assert set(reps.hours) == {24, 168}
    np.testing.assert_allclose(reps.i_ap, reps.ap_original-reps.ap_permuted, atol=1e-15)
    np.testing.assert_allclose(reps.i_brier, reps.brier_permuted-reps.brier_original, atol=1e-15)
    baseline = pd.read_csv(path / "historical_metrics.csv")
    baseline = baseline[baseline.variant.eq("B")].set_index(["scope", "group"])
    for (scope, group), sub in reps.groupby(["scope", "group"]):
        row = baseline.loc[(scope, group)]
        assert sub.n.eq(row.n).all()
        np.testing.assert_allclose(sub.ap_original, row.ap, atol=1e-15)
        np.testing.assert_allclose(sub.brier_original, row.brier, atol=1e-15)
    recorded = json.loads((path / "permutation_execution.json").read_text())
    for key in ["population_preserved", "calendar_preserved", "pairs_preserved", "seed_maps_repeated_exactly", "all_weather_first_rep_predictions_repeated_exactly"]:
        assert recorded[key] is True


def test_protocol_unchanged_and_experiment_has_no_training_calls():
    baseline = json.loads((HERE / "provenance/baseline.json").read_text())
    assert sha(HERE / "protocol.json") == baseline["protocol_sha256"]
    for path in HERE.glob("*.py"):
        tree = ast.parse(path.read_text())
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        assert not {"fit", "fit_transform", "partial_fit", "train"}.intersection(calls), path.name


def test_all_108_columns_and_logical_pairs_are_explicit():
    cols = pd.read_csv(HERE / "effective_columns.csv")
    assert cols.column.tolist() == list(range(108))
    assert cols.iloc[6:57].logical_variable.tolist() == cols.iloc[57:108].logical_variable.tolist()
    assert cols.iloc[:6].component.eq("calendar").all()
    assert cols.iloc[6:57].component.eq("value").all()
    assert cols.iloc[57:108].component.eq("flag").all()
