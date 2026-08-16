"""Tier 2 — EMOS with seasonal and diurnal harmonics.

Once a station has a season or more of history, the residual error left by Tier 1 stops
looking random: it has structure in time of day and time of year. A valley station's
nocturnal cold pool, a coastal station's sea breeze, a snow-covered site's spring albedo
shift — all of these are systematic, periodic, and invisible to a model fitted only on
the forecast value.

We add low-order Fourier terms in day-of-year and hour-of-day to the EMOS mean:

    mu = a + sum_k b_k f_k
         + sum_j [p_j sin(2*pi*j*doy/365.25) + q_j cos(...)]
         + sum_j [u_j sin(2*pi*j*hod/24)     + v_j cos(...)]

Two harmonics each is deliberate. It captures annual and semi-annual cycles (and daily
plus half-daily) without the freedom to chase noise, which matters because this tier
becomes eligible at 90 days — enough data for shape, not enough for wiggle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from wxfuser.config import bucket_for_lead, load_configs, quantiles
from wxfuser.models.distributions import (
    crps_normal,
    crps_truncated_normal,
    quantiles_normal,
    quantiles_truncated_normal,
)
from wxfuser.models.tier1_emos import (
    NONNEGATIVE,
    PRECIP,
    _design,
    _nearest_bucket,
    _tau_days,
    _time_weights,
    clamp_quantiles,
    distribution_for,
    fit_precip,
    predict_precip,
    trim_to_weight,
)

# A cycle can only be fitted from data that covers enough of it. An annual harmonic
# estimated from two months of history is not a seasonal signal — it is an unconstrained
# sine wave through a short arc, and it extrapolates catastrophically the moment it is
# asked to predict outside that arc. This threshold is what stands between Tier 2 being
# an improvement and being far worse than Tier 1.
MIN_SPAN_DAYS_FOR_DOY = 180.0


def training_span_days(valid_time: pd.Series) -> float:
    vt = pd.to_datetime(valid_time)
    if len(vt) == 0:
        return 0.0
    return float((vt.max() - vt.min()).total_seconds() / 86400.0)


def harmonic_features(valid_time: pd.Series, n_doy: int, n_hod: int) -> np.ndarray:
    """Sin/cos pairs for day-of-year and hour-of-day cycles."""
    vt = pd.to_datetime(valid_time)
    doy = vt.dt.dayofyear.to_numpy(dtype=float)
    hod = vt.dt.hour.to_numpy(dtype=float)
    feats = []
    for j in range(1, n_doy + 1):
        ang = 2.0 * np.pi * j * doy / 365.25
        feats += [np.sin(ang), np.cos(ang)]
    for j in range(1, n_hod + 1):
        ang = 2.0 * np.pi * j * hod / 24.0
        feats += [np.sin(ang), np.cos(ang)]
    return np.column_stack(feats) if feats else np.empty((len(vt), 0))


def fit(pairs: pd.DataFrame, variable: str, models: list[str]) -> dict:
    if variable == PRECIP:
        state = fit_precip(pairs, models)
        state["tier"] = "tier2"
        return state

    cfg = load_configs()["tiers"]
    t1, t2 = cfg["tier1"], cfg["tier2"]
    n_doy, n_hod = int(t2["harmonics_doy"]), int(t2["harmonics_hod"])
    truncated = variable in NONNEGATIVE
    crps_fn = crps_truncated_normal if truncated else crps_normal

    df = pairs.dropna(subset=["obs"]).copy()
    if df.empty:
        return {"variable": variable, "models": models, "buckets": {}, "tier": "tier2"}
    ref = pd.to_datetime(df["valid_time"]).max()
    df = df[pd.to_datetime(df["valid_time"]) >= ref - pd.Timedelta(days=int(t1["max_train_days"]))]
    df["_bucket"] = df["lead_h"].map(bucket_for_lead)

    buckets: dict[str, dict] = {}
    for bucket, grp in df.groupby("_bucket"):
        # Harmonics add 2*(n_doy+n_hod) parameters, so demand more data than Tier 1.
        if len(grp) < max(int(t1["min_samples_per_bucket"]) * 2, 120):
            continue
        # Harmonics need a long enough window to see the seasonal cycle, so this tier
        # deliberately trims less aggressively than Tier 1.
        tau = _tau_days(int(grp["lead_h"].median()))
        grp = trim_to_weight(grp, tau * 2.0, ref)

        # Drop the annual harmonics unless this bucket's window actually spans enough of
        # the year to identify them; the diurnal ones need only a day and always stay.
        span = training_span_days(grp["valid_time"])
        bucket_doy = n_doy if span >= MIN_SPAN_DAYS_FOR_DOY else 0
        if bucket_doy == 0 and n_hod == 0:
            continue

        F, s = _design(grp, models)
        H = harmonic_features(grp["valid_time"], bucket_doy, n_hod)
        y = grp["obs"].to_numpy(dtype=float)
        keep = np.isfinite(y) & np.isfinite(F).all(axis=1)
        if keep.sum() < 120:
            continue
        F, s, H, y = F[keep], s[keep], H[keep], y[keep]
        w = _time_weights(grp["valid_time"], tau, ref)[keep]

        n, k = F.shape
        h = H.shape[1]
        X = np.column_stack([np.ones(n), F, H])
        try:
            beta0 = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        sd0 = max(float(np.sqrt(np.average((y - X @ beta0) ** 2, weights=w))), 1e-3)
        theta0 = np.concatenate([beta0, [np.log(sd0), 0.0]])

        def objective(theta: np.ndarray, X=X, s=s, y=y, w=w, k=k, h=h) -> float:
            mu = X @ theta[: 1 + k + h]
            sigma = np.exp(np.clip(theta[1 + k + h] + theta[2 + k + h] * s, -8.0, 8.0))
            return float(np.average(crps_fn(y, mu, sigma), weights=w))

        bounds = [(None, None)] * (1 + k + h) + [(-8.0, 8.0), (0.0, 5.0)]
        try:
            res = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 600})
        except Exception:  # noqa: BLE001
            continue
        if not np.all(np.isfinite(res.x)):
            continue
        th = res.x
        buckets[bucket] = {
            "a": float(th[0]),
            "b": [float(v) for v in th[1 : 1 + k]],
            "h": [float(v) for v in th[1 + k : 1 + k + h]],
            "c": float(th[1 + k + h]),
            "d": float(th[2 + k + h]),
            "n_train": int(len(y)),
            "tau_days": tau,
            "crps_train": float(res.fun),
            # Stored per bucket because it varies: short leads often have a long archive
            # while long leads only have the lead-resolved window.
            "harmonics": {"doy": bucket_doy, "hod": n_hod},
            "span_days": round(span, 1),
        }

    return {
        "variable": variable,
        "models": models,
        "dist": distribution_for(variable),
        "harmonics": {"doy": n_doy, "hod": n_hod},
        "buckets": buckets,
        "tier": "tier2",
        "train_end": str(ref),
    }


def predict(state: dict, fc: pd.DataFrame, models: list[str]) -> dict:
    if state.get("variable") == PRECIP:
        return predict_precip(state, fc, models)

    buckets = state.get("buckets", {})
    if not buckets:
        return {"calibrated": False}
    default_harm = state.get("harmonics", {"doy": 2, "hod": 2})
    truncated = state.get("dist") == "truncated_normal"

    F, s = _design(fc, models)
    n = len(fc)
    mu = np.full(n, np.nan)
    sigma = np.full(n, np.nan)

    # Each bucket may have been fitted with a different harmonic set, so the design
    # matrix is built per distinct set rather than once for the whole frame.
    harm_cache: dict[tuple[int, int], np.ndarray] = {}

    def features(doy: int, hod: int) -> np.ndarray:
        key = (doy, hod)
        if key not in harm_cache:
            harm_cache[key] = harmonic_features(fc["valid_time"], doy, hod)
        return harm_cache[key]

    for i, lead in enumerate(fc["lead_h"]):
        label = bucket_for_lead(int(lead))
        par = buckets.get(label) or _nearest_bucket(buckets, label)
        if not par:
            continue
        b = np.array(par["b"], dtype=float)
        hcoef = np.array(par.get("h", []), dtype=float)
        ph = par.get("harmonics", default_harm)
        H = features(int(ph.get("doy", 0)), int(ph.get("hod", 0)))
        if len(b) != F.shape[1] or len(hcoef) != H.shape[1]:
            continue
        mu[i] = par["a"] + float(F[i] @ b) + float(H[i] @ hcoef)
        sigma[i] = np.exp(np.clip(par["c"] + par["d"] * s[i], -8.0, 8.0))

    ok = np.isfinite(mu) & np.isfinite(sigma)
    if not ok.any():
        return {"calibrated": False}
    mu = np.where(ok, mu, np.nanmean(F, axis=1))
    sigma = np.where(ok, sigma, np.nanmedian(sigma[ok]))
    qfun = quantiles_truncated_normal if truncated else quantiles_normal
    out = dict(qfun(mu, sigma, quantiles()))
    out = clamp_quantiles(out, state.get("variable", ""))
    out["calibrated"] = True
    return out
