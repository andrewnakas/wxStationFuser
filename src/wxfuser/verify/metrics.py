"""Scores and calibration diagnostics.

The claim this project makes — "more accurate than the raw model" — is only worth
anything if it is measured properly, so the verification is deliberately strict:

  * **CRPS** is the headline score (Gneiting & Raftery 2007). It is proper, so a forecast
    cannot improve it by lying about its uncertainty, and it reduces to absolute error
    for a deterministic forecast, which is what makes raw-model comparison fair.
  * **CRPSS** is reported against two baselines: the best raw model *and* climatology.
    Beating climatology is table stakes; beating the raw model is the actual product.
  * **PIT histograms** catch the failure a score alone hides — a forecast can win on CRPS
    while being systematically overconfident.
  * **Block bootstrap** gives a confidence interval on the gain. Consecutive hours are
    correlated, so resampling individual hours would report a falsely tight interval;
    we resample multi-day blocks instead and only claim a win when the interval excludes
    zero.

Quantile-based CRPS uses the pinball identity: the mean pinball loss over a quantile grid
approximates CRPS, which lets us score every tier the same way regardless of whether it
produced a parametric distribution or a set of quantiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wxfuser.config import load_configs, quantiles


def pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
    d = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    return np.maximum(q * d, (q - 1.0) * d)


# Quadrature grid for CRPS = 2 * integral of pinball loss over quantile levels.
# The density matters more than it looks. Evaluating the integral on only the five
# published levels misses the tail contributions entirely, and — critically — the size of
# that error depends on how dispersed the forecast is: it is *exact* for a point forecast
# and understates a calibrated Gaussian by 12%. Since the raw model enters the comparison
# as a point forecast, scoring on five levels flattered every fused-vs-raw claim this
# project makes by roughly 12 percentage points. A 99-level grid brings the error to about
# +1% and, being an overestimate, now errs against our own claim rather than for it.
DENSE_LEVELS = np.round(np.arange(0.01, 0.995, 0.01), 4)


def _quantile_matrix(quantile_preds: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """Sorted (levels, values) from a qNN-keyed prediction dict."""
    items = []
    for key, vals in quantile_preds.items():
        if not (isinstance(key, str) and key.startswith("q") and key[1:].isdigit()):
            continue
        items.append((int(key[1:]) / 100.0, np.asarray(vals, dtype=float)))
    if len(items) < 2:
        return None
    items.sort(key=lambda kv: kv[0])
    levels = np.array([lv for lv, _ in items])
    values = np.column_stack([v for _, v in items])
    return levels, values


def densify_quantiles(
    levels: np.ndarray, values: np.ndarray, dense: np.ndarray = DENSE_LEVELS
) -> np.ndarray:
    """Interpolate a coarse quantile function onto a dense level grid.

    Between the fitted levels this is linear interpolation of the quantile function.
    Outside them the tails are extended with the slope of the outermost fitted interval,
    which is what stops the estimator from silently truncating the distribution — the
    tails are exactly where the missing CRPS mass lives. A degenerate (point) forecast
    interpolates and extends to the same constant, so its CRPS stays exactly the absolute
    error and the raw-model baseline is unaffected.
    """
    n = values.shape[0]
    out = np.empty((n, len(dense)), dtype=float)

    lo_slope = (values[:, 1] - values[:, 0]) / max(levels[1] - levels[0], 1e-9)
    hi_slope = (values[:, -1] - values[:, -2]) / max(levels[-1] - levels[-2], 1e-9)

    for j, d in enumerate(dense):
        if d <= levels[0]:
            out[:, j] = values[:, 0] + lo_slope * (d - levels[0])
        elif d >= levels[-1]:
            out[:, j] = values[:, -1] + hi_slope * (d - levels[-1])
        else:
            i = int(np.searchsorted(levels, d, side="right")) - 1
            w = (d - levels[i]) / (levels[i + 1] - levels[i])
            out[:, j] = values[:, i] * (1.0 - w) + values[:, i + 1] * w
    return out


def crps_from_quantiles(
    y: np.ndarray, quantile_preds: dict[str, np.ndarray], *, dense: bool = True
) -> np.ndarray:
    """CRPS estimated from a quantile forecast, via 2 x mean pinball loss.

    ``dense=True`` first interpolates the published quantiles onto a fine grid so the
    integral covers the tails. Every forecast in the system — fused and raw — is scored
    through this one function, so the comparison stays fair even though the estimator is
    not perfectly exact.
    """
    y = np.asarray(y, dtype=float)
    parsed = _quantile_matrix(quantile_preds)
    if parsed is None:
        return np.full(len(np.atleast_1d(y)), np.nan)
    levels, values = parsed

    if dense:
        grid = DENSE_LEVELS
        mat = densify_quantiles(levels, values, grid)
    else:
        grid, mat = levels, values

    d = y[:, None] - mat
    losses = np.maximum(grid[None, :] * d, (grid[None, :] - 1.0) * d)
    return 2.0 * losses.mean(axis=1)


def crpss(crps_forecast: float, crps_reference: float) -> float:
    """Skill score: 1 - CRPS_forecast / CRPS_reference. Positive means better."""
    if not np.isfinite(crps_reference) or crps_reference <= 0:
        return float("nan")
    return float(1.0 - crps_forecast / crps_reference)


def pit_values(y: np.ndarray, quantile_preds: dict[str, np.ndarray]) -> np.ndarray:
    """Probability integral transform, estimated from the quantile grid.

    For each observation, the fraction of predicted quantile levels it falls above —
    a discrete stand-in for F(y). Uniform PIT means calibrated; a U shape means the
    forecast is overconfident, a hump means it is too timid.
    """
    qs = quantiles()
    keys = [f"q{int(q * 100):02d}" for q in qs if f"q{int(q * 100):02d}" in quantile_preds]
    if not keys:
        return np.array([])
    mat = np.column_stack([quantile_preds[k] for k in keys])
    levels = np.array([q for q in qs if f"q{int(q * 100):02d}" in quantile_preds])
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    below = (mat <= y).astype(float)
    # Interpolate the level at which the predicted CDF crosses the observation.
    idx = below.sum(axis=1).astype(int)
    out = np.empty(len(y))
    for i, k in enumerate(idx):
        if k == 0:
            out[i] = levels[0] * 0.5
        elif k >= len(levels):
            out[i] = (levels[-1] + 1.0) / 2.0
        else:
            lo, hi = mat[i, k - 1], mat[i, k]
            frac = 0.5 if hi <= lo else float(np.clip((y[i, 0] - lo) / (hi - lo), 0, 1))
            out[i] = levels[k - 1] + frac * (levels[k] - levels[k - 1])
    return out


def pit_histogram(pit: np.ndarray, bins: int | None = None) -> list[int]:
    if len(pit) == 0:
        return []
    bins = bins or int(load_configs()["tiers"]["verify"]["pit_bins"])
    counts, _ = np.histogram(np.clip(pit, 0, 1), bins=bins, range=(0.0, 1.0))
    return [int(c) for c in counts]


def interval_coverage(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[float, float]:
    """Empirical coverage and mean width of a predictive interval.

    Reported together on purpose: coverage alone rewards uselessly wide intervals, so
    sharpness is shown next to it (maximise sharpness subject to calibration).
    """
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    if not ok.any():
        return float("nan"), float("nan")
    inside = (y[ok] >= lower[ok]) & (y[ok] <= upper[ok])
    return float(inside.mean()), float(np.mean(upper[ok] - lower[ok]))


def brier_score(y: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    """Brier score for the event y > threshold."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & np.isfinite(prob)
    if not ok.any():
        return float("nan")
    event = (y[ok] > threshold).astype(float)
    return float(np.mean((prob[ok] - event) ** 2))


def reliability_bins(
    y: np.ndarray, prob: np.ndarray, threshold: float, n_bins: int = 10
) -> list[dict]:
    """Forecast-probability vs observed-frequency pairs for a reliability diagram."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & np.isfinite(prob)
    if not ok.any():
        return []
    y, prob = y[ok], np.clip(prob[ok], 0, 1)
    event = (y > threshold).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        sel = (prob >= edges[i]) & (prob < edges[i + 1] if i < n_bins - 1 else prob <= 1.0)
        if sel.sum() == 0:
            continue
        out.append(
            {
                "bin_lower": float(edges[i]),
                "bin_upper": float(edges[i + 1]),
                "forecast_prob": float(prob[sel].mean()),
                "observed_freq": float(event[sel].mean()),
                "count": int(sel.sum()),
            }
        )
    return out


def prob_exceed_from_quantiles(quantile_preds: dict[str, np.ndarray], threshold: float) -> np.ndarray:
    """P(Y > threshold) read off the quantile grid by interpolating the predicted CDF."""
    qs = quantiles()
    keys = [f"q{int(q * 100):02d}" for q in qs if f"q{int(q * 100):02d}" in quantile_preds]
    if not keys:
        return np.array([])
    mat = np.column_stack([quantile_preds[k] for k in keys])
    levels = np.array([q for q in qs if f"q{int(q * 100):02d}" in quantile_preds])
    out = np.empty(len(mat))
    for i in range(len(mat)):
        row = mat[i]
        if threshold <= row[0]:
            out[i] = 1.0 - levels[0] * 0.5
        elif threshold >= row[-1]:
            out[i] = (1.0 - levels[-1]) * 0.5
        else:
            cdf = float(np.interp(threshold, row, levels))
            out[i] = 1.0 - cdf
    return np.clip(out, 0.0, 1.0)


def block_bootstrap_ci(
    daily_a: pd.Series,
    daily_b: pd.Series,
    *,
    weights: pd.Series | None = None,
    n_resamples: int | None = None,
    block_days: int | None = None,
    ci: float | None = None,
    seed: int = 12345,
) -> tuple[float, float, float]:
    """Confidence interval on the skill difference between two forecasts.

    Takes per-day mean scores for both forecasts, resamples contiguous multi-day blocks to
    respect serial correlation, and returns (skill score, lower, upper), where the skill
    score 1 - A/B reads directly as "percent better".

    ``weights`` should be each day's row count. Without it every day counts equally while
    the published point estimate averages over rows, so the interval would be centred on a
    slightly different statistic than the number it is quoted against — enough, in
    marginal cases, for the interval and the headline to disagree about the sign.
    """
    cfg = load_configs()["tiers"]["verify"]["bootstrap"]
    n_resamples = n_resamples or int(cfg["n_resamples"])
    block_days = block_days or int(cfg["block_days"])
    ci = ci or float(cfg["ci"])

    frame = {"a": daily_a, "b": daily_b}
    if weights is not None:
        frame["w"] = weights
    joined = pd.concat(frame, axis=1).dropna()
    if len(joined) < 3:
        return float("nan"), float("nan"), float("nan")

    a = joined["a"].to_numpy(dtype=float)
    b = joined["b"].to_numpy(dtype=float)
    w = joined["w"].to_numpy(dtype=float) if weights is not None else np.ones(len(a))
    n = len(a)

    def skill(idx: np.ndarray) -> float:
        wi = w[idx]
        tot = wi.sum()
        if tot <= 0:
            return np.nan
        bm = float((b[idx] * wi).sum() / tot)
        if bm <= 0:
            return np.nan
        return 1.0 - float((a[idx] * wi).sum() / tot) / bm

    point = skill(np.arange(n))

    block_days = max(1, min(block_days, n))
    n_blocks = max(1, int(np.ceil(n / block_days)))
    starts_pool = np.arange(0, max(1, n - block_days + 1))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for r in range(n_resamples):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block_days, n)) for s in starts])[:n]
        stats[r] = skill(idx)
    stats = stats[np.isfinite(stats)]
    if len(stats) == 0:
        return point, float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    return point, float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


# Observations this close in time to the target are excluded from its climatology.
# Without this the "climatology" for an hour is built partly from that very hour and its
# immediate neighbours — a hindsight forecast no real system could issue, which makes the
# baseline unbeatable and the reported skill against it meaningless.
CLIMATOLOGY_EXCLUSION_DAYS = 3.0


def climatology_quantiles(
    obs: pd.DataFrame, variable_col: str, target_times: pd.Series
) -> dict[str, np.ndarray]:
    """Hour-of-day x day-of-year empirical climatology, the second baseline.

    For each target hour we take observations from the same hour of day within a window of
    days-of-year across all available years, and use their empirical quantiles. This is
    the "what would you predict knowing only the calendar" forecast; a system that cannot
    beat it is not doing anything.

    Observations within a few days of the target are excluded, so the baseline never
    contains the answer it is being asked to predict.
    """
    window = int(load_configs()["tiers"]["verify"]["climatology_doy_window"])
    qs = quantiles()
    df = obs.dropna(subset=[variable_col]).copy()
    out = {f"q{int(q * 100):02d}": np.full(len(target_times), np.nan) for q in qs}
    if df.empty:
        return out

    times = pd.to_datetime(df["valid_time"])
    df["_hod"] = times.dt.hour
    df["_doy"] = times.dt.dayofyear
    values = df[variable_col].to_numpy(dtype=float)
    hod = df["_hod"].to_numpy()
    doy = df["_doy"].to_numpy()
    abs_days = times.to_numpy().astype("datetime64[s]").astype("float64") / 86400.0

    tt = pd.to_datetime(target_times)
    target_days = tt.to_numpy().astype("datetime64[s]").astype("float64") / 86400.0

    for i, ts in enumerate(tt):
        h, d = ts.hour, ts.dayofyear
        # Circular day-of-year distance so late December matches early January.
        dd = np.abs(doy - d)
        dd = np.minimum(dd, 365 - dd)
        far_enough = np.abs(abs_days - target_days[i]) > CLIMATOLOGY_EXCLUSION_DAYS
        sel = (hod == h) & (dd <= window) & far_enough
        if sel.sum() < 10:
            sel = (dd <= window) & far_enough
        if sel.sum() < 5:
            continue
        vals = values[sel]
        for q in qs:
            out[f"q{int(q * 100):02d}"][i] = float(np.quantile(vals, q))
    return out
