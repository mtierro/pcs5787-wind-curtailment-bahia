"""Scientific invariants, including cancellations and incomplete temporal blocks."""
import numpy as np
import pandas as pd
import pytest
from core import block_map, perturb, paired_columns, importance_parts, scores, degradations, local_examples


@pytest.mark.parametrize("hours", [24, 168])
def test_blocks_are_hourly_month_stratified_bijections_with_fixed_tail(hours):
    times = pd.date_range("2026-04-01", "2026-05-04 05:00", freq="h", tz="America/Bahia").as_unit("us")
    mapping, diagnostic = block_map(times, hours, (57872026, "test", "origin", 0))
    repeated, _ = block_map(times, hours, (57872026, "test", "origin", 0))
    assert np.array_equal(mapping, repeated)
    assert np.array_equal(np.sort(mapping), np.arange(len(times)))
    assert np.array_equal(times.month, times[mapping].month)
    for month in np.unique(times.month):
        pos = np.flatnonzero(times.month == month)
        complete = len(pos) // hours * hours
        for start in range(0, complete, hours):
            assert (np.diff(mapping[pos[start:start+hours]]) == 1).all()
        assert np.array_equal(mapping[pos[complete:]], pos[complete:])
    assert sum(d["n_hours"] for d in diagnostic) == len(times)
    # Applying the same time map to a complete region grid never changes regions.
    regions = np.tile(np.arange(8), len(times))
    rowmap = (mapping[:, None] * 8 + np.arange(8)).ravel()
    assert np.array_equal(regions[rowmap], regions)


def test_short_segment_identity_and_missing_hour_rejected():
    times = pd.date_range("2026-01-01", periods=23, freq="h", tz="America/Bahia")
    mapping, _ = block_map(times, 24, (1, "test", "origin", 0))
    assert np.array_equal(mapping, np.arange(23))
    with pytest.raises(AssertionError):
        block_map(times.delete(5), 24, (1, "test", "origin", 0))


def test_value_flag_pairing_and_calendar_invariance_with_missing_donors():
    z = np.ones((4, 108))
    z[:, 3] = np.nan
    z[:, 57:] = 0
    donor = z.copy()
    donor[2, 6] = np.nan
    donor[2, 57] = 1
    donor[:, 0] = 99  # must not reach calendar
    result = perturb(z, donor, [0])
    assert len(result) == len(z)
    assert np.isnan(result[2, 6]) and result[2, 57] == 1
    assert np.array_equal(result[:, :6], z[:, :6], equal_nan=True)
    assert np.array_equal(paired_columns([0, 50]), [6, 56, 57, 107])
    assert np.array_equal(result[:, 7:57], z[:, 7:57])


def test_abs_sum_is_not_sum_of_abs():
    value = np.array([[2., -3.], [-2., 3.]])
    flag = -value
    stats = importance_parts(value, flag)
    assert np.array_equal(stats["mean_abs_joint"], [0, 0])
    assert np.array_equal(stats["sum_component_mean_abs"], [4, 6])
    assert np.array_equal((value + flag).sum(axis=1), value.sum(axis=1) + flag.sum(axis=1))


def test_degradation_signs_and_pooled_ap():
    y = np.array([0, 1, 0, 1])
    p = np.array([.1, .9, .2, .8])
    ap, br = degradations(y, p, 1-p)
    assert ap > 0 and br > 0
    ap, br = degradations(y, 1-p, p)
    assert ap < 0 and br < 0
    p = np.array([.7, .8, .1, .2])
    assert scores(y, p)[0] != np.mean([scores(y[:2], p[:2])[0], scores(y[2:], p[2:])[0]])


def test_local_selection_uses_median_and_stable_key_tiebreak():
    obs = pd.DataFrame(dict(origin=["one"]*6, p=[.1, .2, .3, .6, .7, .8],
                            threshold=[.5]*6, y=[0, 0, 0, 1, 1, 0],
                            time_local=pd.date_range("2026-01-01", periods=6, freq="h"),
                            region=["A"]*6, row_id=range(6)))
    selected = local_examples(obs).set_index("cell")
    assert selected.loc["TN", "row_id"] == 1
    assert selected.loc["TP", "row_id"] == 3
    assert selected.loc["FP", "row_id"] == 5
    assert "FN" not in selected.index
