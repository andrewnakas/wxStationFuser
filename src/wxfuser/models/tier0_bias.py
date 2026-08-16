"""Tier 0 — adaptive bias correction, the method that works on day one.

A decaying-average update of the mean error, kept separately per lead bucket and hour of
day. This is the fixed-gain limit of the Kalman-filter bias schemes in Delle Monache et
al. (2011): it needs no matrix algebra, converges within a couple of weeks of pairs, and
removes the systematic component of model error, which at a typical station is a large
share of its total error. It cannot model the random component, so the spread comes from
the empirical distribution of recent residuals rather than from any fitted parameter.

Keeping the bias per hour-of-day matters because model error at a station is usually
diurnal: a valley station that decouples at night is warm-biased at 05Z and unbiased at
18Z, and a single scalar would split the difference and help neither.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wxfuser.config import bucket_for_lead, load_configs, quantiles

# Variables with a hard floor at zero. Their quantiles are clamped, and precipitation
# additionally gets no additive bias: shifting a variable that is zero most of the time
# by its mean error just smears the dry hours, which costs more than the bias correction
# saves. Precipitation is properly handled by the two-part model in Tier 1.
NONNEGATIVE = {"wind_speed_ms", "wind_gust_ms", "precip_1h_mm", "rh_pct"}
NO_ADDITIVE_BIAS = {"precip_1h_mm"}

VARIABLE_CEILING = {"rh_pct": 100.0}


def fit(pairs: pd.DataFrame, variable: str) -> dict:
    """Fit bias and residual quantiles from a paired archive.

    ``pairs`` needs columns: valid_time, lead_h, fc_mean (multi-model mean), obs.
    Applied as a single pass over time-ordered pairs so the stored state is exactly what
    an online update would have produced.
    """
    cfg = load_configs()["tiers"]["tier0"]
    w = float(cfg["decay_w"])
    min_cell = int(cfg["min_cell_samples"])
    resid_window = int(cfg["residual_window"])
    min_resid = int(cfg["min_residuals_for_bands"])

    df = pairs.dropna(subset=["fc_mean", "obs"]).sort_values("valid_time")
    if df.empty:
        return {"variable": variable, "n_train": 0, "bias": {}, "bias_lead": {}, "resid_q": {}}

    bucket = df["lead_h"].map(bucket_for_lead)
    hod = pd.to_datetime(df["valid_time"]).dt.hour
    error = (df["fc_mean"] - df["obs"]).to_numpy(dtype=float)

    bias_cell: dict[str, float] = {}
    count_cell: dict[str, int] = {}
    bias_lead: dict[str, float] = {}
    count_lead: dict[str, int] = {}

    cell_keys = (bucket.astype(str) + "|" + hod.astype(str)).to_numpy()
    lead_keys = bucket.astype(str).to_numpy()

    if variable in NO_ADDITIVE_BIAS:
        error = np.zeros_like(error)

    for ck, lk, err in zip(cell_keys, lead_keys, error):
        bias_cell[ck] = err if ck not in bias_cell else (1 - w) * bias_cell[ck] + w * err
        count_cell[ck] = count_cell.get(ck, 0) + 1
        bias_lead[lk] = err if lk not in bias_lead else (1 - w) * bias_lead[lk] + w * err
        count_lead[lk] = count_lead.get(lk, 0) + 1

    # Residuals after correction, for the empirical predictive bands.
    corrected = df["fc_mean"].to_numpy(dtype=float) - np.array(
        [
            bias_cell[ck] if count_cell.get(ck, 0) >= min_cell else bias_lead.get(lk, 0.0)
            for ck, lk in zip(cell_keys, lead_keys)
        ]
    )
    resid = df["obs"].to_numpy(dtype=float) - corrected

    qs = quantiles()
    resid_q: dict[str, dict[str, float]] = {}
    for lk in sorted(set(lead_keys)):
        sel = resid[lead_keys == lk][-resid_window:]
        if len(sel) >= min_resid:
            resid_q[lk] = {f"{q}": float(np.quantile(sel, q)) for q in qs}

    # A pooled fallback across all lead buckets. Without it, a bucket with too few
    # residuals of its own fell back to offsets of zero, which is not a wide interval —
    # it is a claim of certainty, published as q05 == q50 == q95. Borrowing the pooled
    # spread is approximate (it ignores how error grows with lead) but it is honest about
    # there being uncertainty at all.
    pooled = resid[-max(resid_window * 3, 120):]
    resid_q_pooled = (
        {f"{q}": float(np.quantile(pooled, q)) for q in qs} if len(pooled) >= min_resid else {}
    )

    return {
        "variable": variable,
        "n_train": int(len(df)),
        "decay_w": w,
        "nonnegative": variable in NONNEGATIVE,
        "ceiling": VARIABLE_CEILING.get(variable),
        "bias": {k: float(v) for k, v in bias_cell.items()},
        "bias_counts": {k: int(v) for k, v in count_cell.items()},
        "bias_lead": {k: float(v) for k, v in bias_lead.items()},
        "resid_q": resid_q,
        "resid_q_pooled": resid_q_pooled,
        "train_end": str(df["valid_time"].max()),
    }


def predict(state: dict, fc_mean: np.ndarray, lead_h: np.ndarray, valid_time: pd.Series) -> dict:
    """Apply the stored bias and residual bands to new forecasts.

    Returns a dict of quantile arrays plus ``calibrated``: False means we corrected the
    median but have too few residuals to claim a spread, and the caller must not present
    bands. Saying so is better than inventing an interval from four samples.
    """
    cfg = load_configs()["tiers"]["tier0"]
    min_cell = int(cfg["min_cell_samples"])

    bias_cell = state.get("bias", {})
    counts = state.get("bias_counts", {})
    bias_lead = state.get("bias_lead", {})
    resid_q = state.get("resid_q", {})
    resid_pooled = state.get("resid_q_pooled", {})

    buckets = [bucket_for_lead(int(h)) for h in lead_h]
    hours = pd.to_datetime(valid_time).dt.hour.to_numpy()

    bias = np.array(
        [
            bias_cell[f"{b}|{h}"]
            if counts.get(f"{b}|{h}", 0) >= min_cell and f"{b}|{h}" in bias_cell
            else bias_lead.get(b, 0.0)
            for b, h in zip(buckets, hours)
        ]
    )
    median = np.asarray(fc_mean, dtype=float) - bias

    qs = quantiles()
    out: dict[str, np.ndarray] = {}
    calibrated = bool(resid_q) or bool(resid_pooled)
    floor = 0.0 if state.get("nonnegative") else None
    ceiling = state.get("ceiling")

    for q in qs:
        if calibrated:
            # Per-bucket residuals where we have them, else the pooled spread. Falling
            # through to a zero offset would publish a zero-width interval — a claim of
            # certainty — for exactly the leads we know least about.
            offsets = np.array(
                [
                    resid_q.get(b, {}).get(f"{q}", resid_pooled.get(f"{q}", np.nan))
                    for b in buckets
                ]
            )
            vals = median + np.nan_to_num(offsets, nan=0.0)
        else:
            vals = median.copy()
        if floor is not None:
            vals = np.maximum(vals, floor)
        if ceiling is not None:
            vals = np.minimum(vals, ceiling)
        out[f"q{int(q * 100):02d}"] = vals

    # Clamping can reorder quantiles at the floor; enforce monotonicity so the published
    # bands are never inverted.
    keys = [f"q{int(q * 100):02d}" for q in qs]
    stacked = np.sort(np.column_stack([out[k] for k in keys]), axis=1)
    for i, k in enumerate(keys):
        out[k] = stacked[:, i]

    out["calibrated"] = calibrated
    return out
