"""Tier 3 — gradient-boosted quantile regression.

Once a station has a year or more of paired history, there is enough data to drop the
linear-and-Gaussian assumptions of EMOS and let the model find the error structure itself:
interactions between lead time, time of day, season, model disagreement, and the forecast
value. That is where quantile regression forests and boosted quantile methods start to
beat EMOS in the literature, particularly for wind and precipitation, whose errors are
skewed and heteroscedastic in ways a Gaussian cannot express.

Two design decisions are doing most of the work here:

  * **Pool across lead times.** EMOS fits each lead bucket separately, which is fine for
    seven parameters but starves a tree ensemble. Training one model over all leads with
    ``lead_h`` as a feature multiplies the sample size by the number of buckets and lets
    the trees learn how error grows with lead, rather than being told.
  * **Fit each quantile directly.** The pinball objective at five quantile levels gives a
    predictive distribution with no distributional assumption at all — the shape is
    whatever the data says, including asymmetry that EMOS structurally cannot represent.

Quantile crossing (the 25th predicted above the 75th) is possible because the levels are
fitted independently; it is resolved by sorting at prediction time, which is the standard
remedy and cannot make any individual quantile worse in expectation.

LightGBM is an optional dependency. Without it this tier reports itself unavailable and
selection simply never considers it, rather than the pipeline failing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wxfuser.config import load_configs, quantiles
from wxfuser.models.tier1_emos import clamp_quantiles

FEATURES_BASE = ["fc_mean", "fc_spread", "lead_h", "sin_doy", "cos_doy", "sin_hod", "cos_hod"]


def available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


def build_features(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Feature matrix: every model's forecast, their agreement, and calendar position."""
    out = pd.DataFrame(index=df.index)
    for m in models:
        col = f"fc_{m}"
        out[col] = df[col] if col in df else np.nan

    model_cols = [f"fc_{m}" for m in models]
    with np.errstate(invalid="ignore"):
        out["fc_mean"] = out[model_cols].mean(axis=1, skipna=True)
        out["fc_spread"] = out[model_cols].std(axis=1, skipna=True, ddof=0).fillna(0.0)

    vt = pd.to_datetime(df["valid_time"])
    doy = vt.dt.dayofyear.to_numpy(dtype=float)
    hod = vt.dt.hour.to_numpy(dtype=float)
    out["lead_h"] = df["lead_h"].to_numpy(dtype=float)
    out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    out["sin_hod"] = np.sin(2 * np.pi * hod / 24.0)
    out["cos_hod"] = np.cos(2 * np.pi * hod / 24.0)
    return out


def feature_names(models: list[str]) -> list[str]:
    return [f"fc_{m}" for m in models] + FEATURES_BASE


def fit(pairs: pd.DataFrame, variable: str, models: list[str]) -> dict:
    """Train one booster per quantile level, pooled across lead times."""
    if not available():
        return {"variable": variable, "models": models, "boosters": {},
                "unavailable": "lightgbm is not installed"}

    import lightgbm as lgb

    cfg = load_configs()["tiers"]["tier3"]
    params = dict(cfg["lgbm"])
    early_days = int(cfg.get("early_stopping_days", 60))

    df = pairs.dropna(subset=["obs"]).sort_values("valid_time").reset_index(drop=True)
    if len(df) < 2000:
        return {"variable": variable, "models": models, "boosters": {},
                "unavailable": "not enough paired history"}

    X = build_features(df, models)
    y = df["obs"].to_numpy(dtype=float)
    names = feature_names(models)
    X = X[names]

    # Hold out the most recent block for early stopping — validating on a random split
    # would leak future information into a model whose whole purpose is forecasting.
    ref = pd.to_datetime(df["valid_time"]).max()
    cutoff = ref - pd.Timedelta(days=early_days)
    is_valid = (pd.to_datetime(df["valid_time"]) >= cutoff).to_numpy()
    if is_valid.sum() < 100 or (~is_valid).sum() < 500:
        split = int(len(df) * 0.85)
        is_valid = np.zeros(len(df), dtype=bool)
        is_valid[split:] = True

    boosters: dict[str, str] = {}
    for q in quantiles():
        train_set = lgb.Dataset(X[~is_valid], label=y[~is_valid], feature_name=names)
        valid_set = lgb.Dataset(X[is_valid], label=y[is_valid], reference=train_set)
        booster = lgb.train(
            {
                "objective": "quantile",
                "alpha": q,
                "metric": "quantile",
                "verbosity": -1,
                "learning_rate": params["learning_rate"],
                "num_leaves": params["num_leaves"],
                "min_data_in_leaf": params["min_data_in_leaf"],
                "feature_fraction": params["feature_fraction"],
                "bagging_fraction": params["bagging_fraction"],
                "bagging_freq": params["bagging_freq"],
            },
            train_set,
            num_boost_round=params["n_estimators"],
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        boosters[f"q{int(q * 100):02d}"] = booster.model_to_string()

    return {
        "variable": variable,
        "models": models,
        "features": names,
        "boosters": boosters,
        "n_train": int((~is_valid).sum()),
        "n_valid": int(is_valid.sum()),
        "train_end": str(ref),
        "tier": "tier3",
    }


def predict(state: dict, fc: pd.DataFrame, models: list[str]) -> dict:
    """Predictive quantiles from the stored boosters."""
    boosters = state.get("boosters") or {}
    if not boosters or not available():
        return {"calibrated": False}

    import lightgbm as lgb

    names = state.get("features") or feature_names(models)
    X = build_features(fc, models)
    for n in names:
        if n not in X:
            X[n] = np.nan
    X = X[names]

    out: dict[str, np.ndarray] = {}
    for key, text in boosters.items():
        try:
            booster = lgb.Booster(model_str=text)
            out[key] = booster.predict(X)
        except Exception as exc:  # noqa: BLE001
            print(f"    WARN: tier3 booster {key} failed ({exc})", flush=True)
            return {"calibrated": False}

    # Independently fitted quantiles can cross; sorting is the standard remedy.
    out = clamp_quantiles(out, state.get("variable", ""))
    out["calibrated"] = True
    return out
