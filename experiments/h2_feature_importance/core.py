"""Pure numerical operations; no fitting and no historical writes."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

FAMILIES = {"anchor": list(range(8)), "temporal": list(range(8, 28)),
            "atmospheric": list(range(28, 37)), "multisensor": list(range(37, 51))}
FAMILY_LABELS = {"anchor": "Vento e rajada — estação âncora", "temporal": "Dinâmica e resumos temporais",
                 "atmospheric": "Direção e atmosfera", "multisensor": "Múltiplas estações"}
CALENDAR = ["region", "hour", "weekday", "month", "region_hour", "region_month"]


def seeded_int(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def block_map(times, hours, seed_parts):
    """Map full hourly destination positions to source positions within civil month.

    Identity maps/fixed points are valid draws. A final partial block is fixed.
    Source rows need no known target: no source target is ever used.
    """
    times = pd.DatetimeIndex(times).as_unit("ns")
    assert times.is_unique and times.is_monotonic_increasing
    assert len(times) < 2 or (np.diff(times.asi8) == pd.Timedelta(hours=1).value).all()
    mapping = np.arange(len(times))
    diagnostics = []
    for month in np.unique(times.strftime("%Y-%m")):
        pos = np.flatnonzero(times.strftime("%Y-%m") == month)
        nblocks, tail = divmod(len(pos), hours)
        master, split, origin, rep = seed_parts
        seed = seeded_int(master, split, origin, month, hours, rep)
        order = np.random.default_rng(seed).permutation(nblocks)
        complete = nblocks * hours
        if complete:
            mapping[pos[:complete]] = pos[(order[:, None] * hours + np.arange(hours)).ravel()]
        assert np.array_equal(np.sort(mapping[pos]), pos)
        assert np.array_equal(mapping[pos[complete:]], pos[complete:])
        diagnostics.append(dict(month=month, seed=str(seed), hours=hours, n_hours=len(pos),
                                complete_blocks=nblocks, fixed_tail_hours=tail,
                                moved_hours=int(np.sum(mapping[pos] != pos)),
                                order=",".join(map(str, order))))
    return mapping, diagnostics


def paired_columns(indices):
    indices = np.asarray(indices, dtype=int)
    return np.r_[6 + indices, 57 + indices]


def perturb(z, donor, indices):
    out = z.copy()
    columns = paired_columns(indices)
    out[:, columns] = donor[:, columns]
    assert np.array_equal(out[:, :6], z[:, :6], equal_nan=True)
    assert np.array_equal(out[:, 57:108], (~np.isfinite(out[:, 6:57])).astype(float))
    return out


def importance_parts(value, flag):
    joint = value + flag
    return {"mean_abs_value": np.abs(value).mean(axis=0),
            "mean_abs_flag": np.abs(flag).mean(axis=0),
            "mean_abs_joint": np.abs(joint).mean(axis=0),
            "sum_component_mean_abs": (np.abs(value) + np.abs(flag)).mean(axis=0),
            "mean_signed_joint": joint.mean(axis=0)}


def scores(y, p):
    return float(average_precision_score(y, p)), float(brier_score_loss(y, p))


def degradations(y, baseline, permuted):
    ap0, b0 = scores(y, baseline)
    ap, b = scores(y, permuted)
    return ap0 - ap, b - b0


def local_examples(obs):
    out = []
    for origin, frame in obs.groupby("origin", sort=True):
        pred = frame.p.ge(frame.threshold)
        actual = frame.y.eq(1)
        cells = {"TP": actual & pred, "FP": ~actual & pred,
                 "FN": actual & ~pred, "TN": ~actual & ~pred}
        for cell, mask in cells.items():
            subset = frame.loc[mask].copy()
            if len(subset):
                subset["distance_to_median"] = (subset.p - subset.p.median()).abs()
                chosen = subset.sort_values(["distance_to_median", "time_local", "region"]).iloc[0]
                out.append({**chosen.to_dict(), "cell": cell, "cell_n": len(subset)})
    return pd.DataFrame(out)


def strata(obs, interaction=False):
    yield "global", "all", np.arange(len(obs))
    for col in ["origin", "region"]:
        for value, positions in obs.groupby(col, sort=True).indices.items():
            yield col, str(value), np.asarray(positions)
    if interaction:
        for (origin, region), positions in obs.groupby(["origin", "region"], sort=True).indices.items():
            yield "origin_region", f"{origin}|{region}", np.asarray(positions)
