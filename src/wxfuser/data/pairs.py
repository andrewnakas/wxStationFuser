"""The paired forecast/observation archive — the training set.

One row per (model, valid_time, lead bucket source), carrying both what each model
predicted and what the station actually measured. Everything downstream trains on this
one table.

Two design points worth stating:

  * **Re-pairing a trailing window.** Observations arrive late and get revised; Meteostat
    lags days, METARs get corrected. So each refresh re-pairs the last week rather than
    only appending new hours, and dedupe keeps the newest version of a row.
  * **Lead provenance is kept.** Rows from the seamless historical archive
    (``lead_source="hist"``) are short-lead by construction. Mixing them into a 5-day
    lead bucket would manufacture skill that does not exist, so the bucket assignment
    respects the source and the emitted skill numbers say which is which.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wxfuser.config import variable_map
from wxfuser.data.obs import OBS_FOR_VARIABLE

PAIR_KEY = ["model", "valid_time", "lead_h", "lead_source"]


def pair_columns(variables: list[str]) -> list[str]:
    cols = ["station_id", *PAIR_KEY]
    cols += [f"fc_{v}" for v in variables]
    cols += [f"obs_{v}" for v in variables]
    cols += ["obs_source"]
    return cols


def build_pairs(
    forecasts: pd.DataFrame,
    obs: pd.DataFrame,
    station_id: str,
    variables: list[str],
) -> pd.DataFrame:
    """Join model forecasts to observations on valid_time.

    ``forecasts`` is long form from the openmeteo client (model, valid_time, lead_h,
    lead_source, fc_*); ``obs`` is the normalised observation schema.
    """
    if forecasts.empty or obs.empty:
        return pd.DataFrame(columns=pair_columns(variables))

    fc = forecasts.copy()
    fc["valid_time"] = pd.to_datetime(fc["valid_time"]).dt.floor("h")

    ob = obs.copy()
    ob["valid_time"] = pd.to_datetime(ob["valid_time"]).dt.floor("h")
    keep = {"valid_time": "valid_time", "source": "obs_source"}
    sel = ob[["valid_time", "source"]].copy()
    for var in variables:
        obs_col = OBS_FOR_VARIABLE.get(var)
        if obs_col and obs_col in ob:
            sel[f"obs_{var}"] = pd.to_numeric(ob[obs_col], errors="coerce")
        else:
            sel[f"obs_{var}"] = np.nan
    sel = sel.rename(columns=keep).drop_duplicates("valid_time", keep="last")

    out = fc.merge(sel, on="valid_time", how="inner")
    if out.empty:
        return pd.DataFrame(columns=pair_columns(variables))
    out["station_id"] = station_id
    for var in variables:
        if f"fc_{var}" not in out:
            out[f"fc_{var}"] = np.nan
    # An hour with no observation of any variable teaches nothing.
    obs_cols = [f"obs_{v}" for v in variables]
    out = out.dropna(subset=obs_cols, how="all")
    return out[pair_columns(variables)].reset_index(drop=True)


def merge_archive(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append new pairs, keeping the newest version of any duplicated key."""
    if existing is None or existing.empty:
        combined = new
    elif new is None or new.empty:
        combined = existing
    else:
        combined = pd.concat([existing, new], ignore_index=True)
    if combined.empty:
        return combined
    combined["valid_time"] = pd.to_datetime(combined["valid_time"])
    # A stable sort is load-bearing, not a style choice: `new` was concatenated after
    # `existing`, so keeping the last of each duplicate key is what makes a revised
    # observation win. With an unstable sort, ties reorder arbitrarily and revisions
    # are silently dropped for some rows.
    combined = combined.sort_values("valid_time", kind="stable")
    return combined.drop_duplicates(PAIR_KEY, keep="last").reset_index(drop=True)


def to_wide(pairs: pd.DataFrame, variable: str, models: list[str]) -> pd.DataFrame:
    """Reshape long pairs into the training matrix for one variable.

    Output columns: valid_time, lead_h, lead_source, obs, fc_{model} per model, fc_mean.
    Only hours where the observation exists survive, since an unobserved hour cannot
    contribute to a fit.
    """
    fc_col, obs_col = f"fc_{variable}", f"obs_{variable}"
    if pairs.empty or fc_col not in pairs or obs_col not in pairs:
        return pd.DataFrame(columns=["valid_time", "lead_h", "lead_source", "obs", "fc_mean"])

    df = pairs[["model", "valid_time", "lead_h", "lead_source", fc_col, obs_col]].copy()
    df = df.dropna(subset=[obs_col])
    if df.empty:
        return pd.DataFrame(columns=["valid_time", "lead_h", "lead_source", "obs", "fc_mean"])

    wide = df.pivot_table(
        index=["valid_time", "lead_h", "lead_source"],
        columns="model",
        values=fc_col,
        aggfunc="last",
    )
    wide.columns = [f"fc_{c}" for c in wide.columns]
    obs_series = (
        df.groupby(["valid_time", "lead_h", "lead_source"])[obs_col].last().rename("obs")
    )
    out = wide.join(obs_series, how="inner").reset_index()

    for m in models:
        if f"fc_{m}" not in out:
            out[f"fc_{m}"] = np.nan
    model_cols = [f"fc_{m}" for m in models]
    with np.errstate(invalid="ignore"):
        out["fc_mean"] = out[model_cols].mean(axis=1, skipna=True)
        out["fc_spread"] = out[model_cols].std(axis=1, skipna=True, ddof=0)
    out = out.dropna(subset=["fc_mean", "obs"])
    out["valid_time"] = pd.to_datetime(out["valid_time"])
    return out.sort_values("valid_time").reset_index(drop=True)


def read_archive(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def write_archive(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in out.columns:
        if c.startswith(("fc_", "obs_")) and out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    out.to_parquet(p, index=False, compression="zstd")


def sanitize_station_id(station_id: str) -> str:
    """Filesystem/URL-safe form of a station id (``IEM:DEN`` -> ``IEM_DEN``)."""
    return station_id.replace(":", "_").replace("/", "_").replace(" ", "_")


def om_variable(variable: str) -> str:
    return variable_map()[variable]
