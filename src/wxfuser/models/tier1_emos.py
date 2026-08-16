"""Tier 1 — Ensemble Model Output Statistics, fitted by minimum CRPS.

The workhorse. EMOS (Gneiting, Raftery, Westveld & Goldman 2005) fits a full predictive
distribution whose mean is a linear function of the model forecasts and whose spread is a
function of how much the models disagree:

    mu    = a + sum_k b_k * f_k
    sigma = exp(c + d * s)          s = inter-model standard deviation

and estimates (a, b, c, d) by minimising CRPS directly rather than by least squares. Two
properties make it the right default here:

  * It was designed for small samples. The original paper used 30-45 day rolling
    training windows, which is exactly the regime a freshly enrolled station is in.
  * The spread is *earned*. With several models in play, ``s`` is a real signal — hours
    where GFS and ECMWF disagree are genuinely less predictable — so the intervals widen
    where they should instead of being a constant band.

Training uses exponential time weighting rather than a hard cutoff (Lang et al. 2020):
recent pairs dominate, older ones fade, and no forecast falls off a cliff on a Tuesday.
The window is longer for long leads, where errors evolve more slowly and data is scarcer.

Wind speed and gusts use a normal truncated at zero; precipitation is not a single
distribution at all and is handled by ``fit_precip``/``predict_precip`` below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from wxfuser.config import bucket_for_lead, load_configs, quantiles
from wxfuser.models.distributions import (
    crps_normal,
    crps_truncated_normal,
    quantiles_normal,
    quantiles_truncated_normal,
)

# Variables whose support is [0, inf) and therefore use the truncated normal.
NONNEGATIVE = {"wind_speed_ms", "wind_gust_ms"}
PRECIP = "precip_1h_mm"

# Physically bounded variables. Relative humidity is the one that bites: it is bounded
# on both sides and spends much of its time near a boundary, so an unclamped Gaussian
# routinely predicts 110% and is scored for it.
VARIABLE_BOUNDS = {
    "rh_pct": (0.0, 100.0),
    "wind_speed_ms": (0.0, None),
    "wind_gust_ms": (0.0, None),
    "precip_1h_mm": (0.0, None),
}


def clamp_quantiles(out: dict, variable: str) -> dict:
    """Apply physical bounds and re-sort, so published bands stay ordered and possible."""
    bounds = VARIABLE_BOUNDS.get(variable)
    keys = [f"q{int(q * 100):02d}" for q in quantiles() if f"q{int(q * 100):02d}" in out]
    if not keys:
        return out
    mat = np.column_stack([np.asarray(out[k], dtype=float) for k in keys])
    if bounds:
        lo, hi = bounds
        if lo is not None:
            mat = np.maximum(mat, lo)
        if hi is not None:
            mat = np.minimum(mat, hi)
    mat = np.sort(mat, axis=1)
    for i, k in enumerate(keys):
        out[k] = mat[:, i]
    return out


def distribution_for(variable: str) -> str:
    if variable == PRECIP:
        return "bernoulli_quantile_map"
    return "truncated_normal" if variable in NONNEGATIVE else "normal"


def _tau_days(lead_h: int) -> float:
    cfg = load_configs()["tiers"]["tier1"]
    return float(
        cfg["tau_days_short"] if lead_h <= cfg["tau_switch_lead_h"] else cfg["tau_days_long"]
    )


def _time_weights(valid_time: pd.Series, tau_days: float, ref: pd.Timestamp) -> np.ndarray:
    age_days = (ref - pd.to_datetime(valid_time)).dt.total_seconds().to_numpy() / 86400.0
    return np.exp(-np.maximum(age_days, 0.0) / tau_days)


def trim_to_weight(grp: pd.DataFrame, tau_days: float, ref: pd.Timestamp) -> pd.DataFrame:
    """Drop rows whose time weight is negligible, then subsample if still huge.

    An exponentially weighted fit is dominated by its most recent few time constants;
    a row five taus old carries weight e^-5 and cannot move the answer. Keeping it costs
    real time in the walk-forward evaluation, where the fit is repeated dozens of times.
    """
    cfg = load_configs()["tiers"]["tier1"]
    cutoff = float(cfg.get("weight_cutoff_taus", 5)) * tau_days
    max_rows = int(cfg.get("max_rows_per_bucket", 6000))

    age = (ref - pd.to_datetime(grp["valid_time"])).dt.total_seconds() / 86400.0
    out = grp[age <= cutoff]
    if len(out) < 40:
        out = grp
    if len(out) > max_rows:
        # Keep the most recent rows; they are the ones with weight.
        out = out.sort_values("valid_time").iloc[-max_rows:]
    return out


def _design(df: pd.DataFrame, models: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-model forecast matrix F (n, k) and inter-model spread s (n,).

    Missing model columns are filled with the row mean so one model's outage does not
    discard the whole hour; spread is computed only over the models actually present.
    """
    cols = [f"fc_{m}" for m in models]
    F = np.array(df[cols].to_numpy(dtype=float), copy=True)
    with np.errstate(invalid="ignore"):
        row_mean = np.nanmean(F, axis=1)
        spread = np.nanstd(F, axis=1, ddof=0) if F.shape[1] > 1 else np.zeros(len(F))
    inds = np.where(np.isnan(F))
    F[inds] = np.take(row_mean, inds[0])
    spread = np.nan_to_num(spread, nan=0.0)
    return F, spread


def _fit_bucket(
    F: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    truncated: bool,
) -> dict | None:
    """Minimise weighted CRPS over (a, b_1..b_k, c, d) for one lead bucket."""
    n, k = F.shape
    if n < 5:
        return None
    crps_fn = crps_truncated_normal if truncated else crps_normal

    # Initialise from a weighted least-squares fit; log residual sd seeds c.
    X = np.column_stack([np.ones(n), F])
    W = np.diag(w)
    try:
        beta0 = np.linalg.lstsq(np.sqrt(W) @ X, np.sqrt(w) * y, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta0 = np.concatenate([[float(np.average(y, weights=w))], np.zeros(k)])
    resid0 = y - X @ beta0
    sd0 = max(float(np.sqrt(np.average(resid0**2, weights=w))), 1e-3)
    theta0 = np.concatenate([beta0, [np.log(sd0), 0.0]])

    def objective(theta: np.ndarray) -> float:
        a, b, c, d = theta[0], theta[1 : 1 + k], theta[1 + k], theta[2 + k]
        mu = a + F @ b
        # d >= 0 is enforced by the bounds, so sigma grows with model disagreement.
        sigma = np.exp(np.clip(c + d * s, -8.0, 8.0))
        return float(np.average(crps_fn(y, mu, sigma), weights=w))

    bounds = [(None, None)] * (1 + k) + [(-8.0, 8.0), (0.0, 5.0)]
    try:
        res = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})
    except Exception:  # noqa: BLE001
        return None
    if not np.all(np.isfinite(res.x)):
        return None

    theta = res.x
    return {
        "a": float(theta[0]),
        "b": [float(v) for v in theta[1 : 1 + k]],
        "c": float(theta[1 + k]),
        "d": float(theta[2 + k]),
        "n_train": int(n),
        "crps_train": float(res.fun),
    }


def fit(pairs: pd.DataFrame, variable: str, models: list[str]) -> dict:
    """Fit one EMOS parameter set per lead bucket.

    ``pairs`` needs: valid_time, lead_h, obs, and fc_{model} for each model.
    """
    if variable == PRECIP:
        return fit_precip(pairs, models)

    cfg = load_configs()["tiers"]["tier1"]
    max_days = int(cfg["max_train_days"])
    min_bucket = int(cfg["min_samples_per_bucket"])
    truncated = variable in NONNEGATIVE

    df = pairs.dropna(subset=["obs"]).copy()
    if df.empty:
        return {"variable": variable, "models": models, "buckets": {}}
    ref = pd.to_datetime(df["valid_time"]).max()
    df = df[pd.to_datetime(df["valid_time"]) >= ref - pd.Timedelta(days=max_days)]
    df["_bucket"] = df["lead_h"].map(bucket_for_lead)

    buckets: dict[str, dict] = {}
    for bucket, grp in df.groupby("_bucket"):
        if len(grp) < min_bucket:
            continue
        tau = _tau_days(int(grp["lead_h"].median()))
        grp = trim_to_weight(grp, tau, ref)
        F, s = _design(grp, models)
        y = grp["obs"].to_numpy(dtype=float)
        keep = np.isfinite(y) & np.isfinite(F).all(axis=1)
        if keep.sum() < min_bucket:
            continue
        w = _time_weights(grp["valid_time"], tau, ref)[keep]
        fitted = _fit_bucket(F[keep], s[keep], y[keep], w, truncated=truncated)
        if fitted:
            fitted["tau_days"] = tau
            buckets[bucket] = fitted

    return {
        "variable": variable,
        "models": models,
        "dist": distribution_for(variable),
        "buckets": buckets,
        "train_end": str(ref),
        "n_train": int(len(df)),
    }


def predict(state: dict, fc: pd.DataFrame, models: list[str]) -> dict:
    """Predictive quantiles for new forecasts. ``fc`` needs lead_h + fc_{model} columns."""
    variable = state.get("variable")
    if variable == PRECIP:
        return predict_precip(state, fc, models)

    truncated = state.get("dist") == "truncated_normal"
    buckets = state.get("buckets", {})
    F, s = _design(fc, models)
    n = len(fc)
    mu = np.full(n, np.nan)
    sigma = np.full(n, np.nan)

    bucket_labels = [bucket_for_lead(int(h)) for h in fc["lead_h"]]
    for i, bl in enumerate(bucket_labels):
        par = buckets.get(bl) or _nearest_bucket(buckets, bl)
        if not par:
            continue
        b = np.array(par["b"], dtype=float)
        if len(b) != F.shape[1]:
            continue
        mu[i] = par["a"] + float(F[i] @ b)
        sigma[i] = np.exp(np.clip(par["c"] + par["d"] * s[i], -8.0, 8.0))

    return finalise_quantiles(mu, sigma, F, variable, truncated=truncated)


def finalise_quantiles(
    mu: np.ndarray,
    sigma: np.ndarray,
    F: np.ndarray,
    variable: str,
    *,
    truncated: bool,
) -> dict:
    """Turn fitted (mu, sigma) into published quantiles, handling rows we cannot fit.

    Three cases, deliberately distinguished:

      * a row with fitted parameters gets the fitted distribution;
      * a row whose lead had no fitted bucket falls back to the raw multi-model mean,
        widened to the broadest spread fitted anywhere — an honest "less sure here"
        rather than a confident guess;
      * a row where every model is missing cannot be forecast at all, and is emitted as
        NaN. Inventing a value would put a fabricated number on the site indistinguishable
        from a real one.
    """
    n = len(mu)
    ok = np.isfinite(mu) & np.isfinite(sigma)
    if not ok.any():
        return {"calibrated": False}

    with np.errstate(invalid="ignore"):
        raw_mean = np.nanmean(F, axis=1) if F.size else np.full(n, np.nan)
    fallback_sigma = float(np.nanmax(sigma[ok]))
    mu_f = np.where(ok, mu, raw_mean)
    sigma_f = np.where(ok, sigma, fallback_sigma)

    usable = np.isfinite(mu_f) & np.isfinite(sigma_f)
    safe_mu = np.where(usable, mu_f, 0.0)
    safe_sigma = np.where(usable, sigma_f, 1.0)

    qfun = quantiles_truncated_normal if truncated else quantiles_normal
    out = {k: np.where(usable, v, np.nan) for k, v in qfun(safe_mu, safe_sigma, quantiles()).items()}
    out = clamp_quantiles(out, variable)
    out["calibrated"] = True
    out["n_unforecastable"] = int((~usable).sum())
    return out


def _nearest_bucket(buckets: dict, label: str) -> dict | None:
    """Fall back to the closest fitted bucket when a lead has none of its own.

    Better to use 25-48 h coefficients for a 60 h forecast than to fall all the way back
    to raw model output.
    """
    if not buckets:
        return None
    try:
        target = int(label.split("-")[0])
    except ValueError:
        return None
    best, best_dist = None, None
    for k, v in buckets.items():
        try:
            lo = int(k.split("-")[0])
        except ValueError:
            continue
        d = abs(lo - target)
        if best_dist is None or d < best_dist:
            best, best_dist = v, d
    return best


# --------------------------------------------------------------------------- precipitation


def fit_precip(pairs: pd.DataFrame, models: list[str], threshold: float = 0.1) -> dict:
    """Two-part precipitation model: occurrence, then amount.

    Precipitation is a mixed random variable — a point mass at zero plus a skewed
    positive part — so a single Gaussian is simply the wrong object. We fit:

      * occurrence: logistic regression of P(precip > threshold) on the multi-model mean
        and the fraction of models predicting wet, which is the useful ensemble signal;
      * amount: quantile mapping from predicted positive amounts onto observed positive
        amounts, which corrects the well-known drizzle bias without assuming a family.

    A censored shifted gamma (Scheuerer & Hamill 2015) is the documented upgrade path.
    """
    cfg = load_configs()["tiers"]["tier1"]
    max_days = int(cfg["max_train_days"])
    df = pairs.dropna(subset=["obs"]).copy()
    if df.empty:
        return {"variable": PRECIP, "models": models, "dist": "bernoulli_quantile_map",
                "buckets": {}}
    ref = pd.to_datetime(df["valid_time"]).max()
    df = df[pd.to_datetime(df["valid_time"]) >= ref - pd.Timedelta(days=max_days)]
    df["_bucket"] = df["lead_h"].map(bucket_for_lead)

    buckets: dict[str, dict] = {}
    for bucket, grp in df.groupby("_bucket"):
        F, _ = _design(grp, models)
        y = grp["obs"].to_numpy(dtype=float)
        keep = np.isfinite(y) & np.isfinite(F).all(axis=1)
        if keep.sum() < 40:
            continue
        F, y = F[keep], y[keep]
        fmean = F.mean(axis=1)
        frac_wet = (F > threshold).mean(axis=1)
        wet = (y > threshold).astype(float)

        coef = _fit_logistic(np.column_stack([fmean, frac_wet]), wet)
        # Quantile map on the positive parts only.
        pos_f = np.sort(fmean[fmean > threshold])
        pos_o = np.sort(y[y > threshold])
        grid = np.linspace(0.02, 0.98, 25)
        buckets[bucket] = {
            "logit": coef,
            "qmap_f": [float(v) for v in np.quantile(pos_f, grid)] if len(pos_f) > 5 else [],
            "qmap_o": [float(v) for v in np.quantile(pos_o, grid)] if len(pos_o) > 5 else [],
            "grid": [float(v) for v in grid],
            "threshold": threshold,
            "n_train": int(len(y)),
            "base_rate": float(wet.mean()),
        }

    return {
        "variable": PRECIP,
        "models": models,
        "dist": "bernoulli_quantile_map",
        "buckets": buckets,
        "train_end": str(ref),
        "n_train": int(len(df)),
    }


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> list[float]:
    """Small ridge-penalised logistic fit (intercept + coefficients), no sklearn dependency."""
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])

    def nll(theta: np.ndarray) -> float:
        z = np.clip(Xd @ theta, -30, 30)
        p = expit(z)
        eps = 1e-9
        return float(
            -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            + l2 * float(theta[1:] @ theta[1:])
        )

    theta0 = np.zeros(k + 1)
    base = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
    theta0[0] = np.log(base / (1 - base))
    try:
        res = minimize(nll, theta0, method="L-BFGS-B", options={"maxiter": 300})
        return [float(v) for v in res.x]
    except Exception:  # noqa: BLE001
        return [float(theta0[0])] + [0.0] * k


def predict_precip(state: dict, fc: pd.DataFrame, models: list[str]) -> dict:
    """Occurrence probability plus amount quantiles, with dry hours pinned to zero."""
    buckets = state.get("buckets", {})
    F, _ = _design(fc, models)
    n = len(fc)
    qs = quantiles()
    out = {f"q{int(q * 100):02d}": np.zeros(n) for q in qs}
    p_occ = np.zeros(n)

    labels = [bucket_for_lead(int(h)) for h in fc["lead_h"]]
    for i, bl in enumerate(labels):
        par = buckets.get(bl) or _nearest_bucket(buckets, bl)
        if not par:
            continue
        thr = par.get("threshold", 0.1)
        fmean = float(np.nanmean(F[i]))
        frac_wet = float(np.nanmean(F[i] > thr))
        coef = par.get("logit") or [0.0, 0.0, 0.0]
        p = float(expit(np.clip(coef[0] + coef[1] * fmean + coef[2] * frac_wet, -30, 30)))
        p_occ[i] = p

        qmap_f, qmap_o = par.get("qmap_f") or [], par.get("qmap_o") or []
        amount = fmean
        if qmap_f and qmap_o:
            amount = float(np.interp(fmean, qmap_f, qmap_o))
        for q in qs:
            # The distribution has mass 1-p at zero, so quantile q is zero whenever
            # q <= 1-p; above that we scale the mapped amount by how far into the wet
            # part of the distribution the quantile falls.
            if q <= 1.0 - p:
                out[f"q{int(q * 100):02d}"][i] = 0.0
            else:
                frac = (q - (1.0 - p)) / max(p, 1e-6)
                out[f"q{int(q * 100):02d}"][i] = max(amount * (0.4 + 1.6 * frac), 0.0)

    out["p_occ"] = p_occ
    out["calibrated"] = bool(buckets)
    return out
