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


def crps_from_quantiles(y: np.ndarray, quantile_preds: dict[str, np.ndarray]) -> np.ndarray:
    """Approximate CRPS from a quantile grid (mean pinball loss x 2).

    For a quantile grid Q, CRPS = 2 * mean_q pinball_q. Exact in the limit of a dense
    grid; with our five quantiles it is a consistent estimator that ranks methods
    correctly, which is what selection needs.
    """
    qs = quantiles()
    losses = []
    for q in qs:
        key = f"q{int(q * 100):02d}"
        if key not in quantile_preds:
            continue
        losses.append(pinball_loss(y, quantile_preds[key], q))
    if not losses:
        return np.full(len(np.atleast_1d(y)), np.nan)
    return 2.0 * np.mean(losses, axis=0)


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
    n_resamples: int | None = None,
    block_days: int | None = None,
    ci: float | None = None,
    seed: int = 12345,
) -> tuple[float, float, float]:
    """Confidence interval on the *skill difference* between two forecasts.

    Takes per-day mean scores for both forecasts, resamples multi-day blocks to respect
    serial correlation, and returns (mean skill score, lower, upper). We report the
    skill score 1 - A/B so the interval is directly interpretable as "percent better".
    """
    cfg = load_configs()["tiers"]["verify"]["bootstrap"]
    n_resamples = n_resamples or int(cfg["n_resamples"])
    block_days = block_days or int(cfg["block_days"])
    ci = ci or float(cfg["ci"])

    joined = pd.concat([daily_a.rename("a"), daily_b.rename("b")], axis=1).dropna()
    if len(joined) < 3:
        return float("nan"), float("nan"), float("nan")

    a = joined["a"].to_numpy(dtype=float)
    b = joined["b"].to_numpy(dtype=float)
    n = len(a)
    point = 1.0 - a.mean() / b.mean() if b.mean() > 0 else float("nan")

    n_blocks = max(1, int(np.ceil(n / block_days)))
    starts_pool = np.arange(0, max(1, n - block_days + 1))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for r in range(n_resamples):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block_days, n)) for s in starts])[:n]
        bm = b[idx].mean()
        stats[r] = 1.0 - a[idx].mean() / bm if bm > 0 else np.nan
    stats = stats[np.isfinite(stats)]
    if len(stats) == 0:
        return point, float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    return point, float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def climatology_quantiles(
    obs: pd.DataFrame, variable_col: str, target_times: pd.Series
) -> dict[str, np.ndarray]:
    """Hour-of-day x day-of-year empirical climatology, the second baseline.

    For each target hour we take observations from the same hour of day within a window
    of days-of-year across all available years, and use their empirical quantiles. This
    is the "what would you predict knowing only the calendar" forecast; a system that
    cannot beat it is not doing anything.
    """
    window = int(load_configs()["tiers"]["verify"]["climatology_doy_window"])
    qs = quantiles()
    df = obs.dropna(subset=[variable_col]).copy()
    out = {f"q{int(q * 100):02d}": np.full(len(target_times), np.nan) for q in qs}
    if df.empty:
        return out

    df["_hod"] = pd.to_datetime(df["valid_time"]).dt.hour
    df["_doy"] = pd.to_datetime(df["valid_time"]).dt.dayofyear
    values = df[variable_col].to_numpy(dtype=float)
    hod = df["_hod"].to_numpy()
    doy = df["_doy"].to_numpy()

    tt = pd.to_datetime(target_times)
    for i, ts in enumerate(tt):
        h, d = ts.hour, ts.dayofyear
        # Circular day-of-year distance so late December matches early January.
        dd = np.abs(doy - d)
        dd = np.minimum(dd, 365 - dd)
        sel = (hod == h) & (dd <= window)
        if sel.sum() < 10:
            sel = dd <= window
        if sel.sum() < 5:
            continue
        vals = values[sel]
        for q in qs:
            out[f"q{int(q * 100):02d}"][i] = float(np.quantile(vals, q))
    return out
