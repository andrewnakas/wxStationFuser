"""Derived variables: 24-hour snow and water amounts, built from hourly data on both sides.

Avalanche forecasters reason in 24-hour accumulations, such as "30 cm of new snow by
morning" or "25 mm SWE loading", not hourly rates. Neither side of the fusion serves
these directly:

  * **Models** give hourly (or 3- or 6-hourly) precipitation and temperature. The 24 h
    water amount is a windowed sum within one forecast. New snow additionally needs the
    fraction falling as snow and its density, both taken from temperature: a plain
    physical prior whose station-specific errors are what calibration then removes.
  * **Stations** give snow depth and snow water equivalent (SNOTEL's pillow and depth
    sensor). The 24 h change in each is the observed amount. Both are median-smoothed
    first, and gains within one reporting step are treated as none: the sensors flicker
    between adjacent readings, which read raw would put phantom new snow in every calm
    hour, including all summer over bare ground.

Every derived value is labelled at the *end* of its window, so ``hn24_cm`` at 14:00 is
the new snow from 14:00 the previous day to 14:00, published for every hour. A window
missing any part of its 24 hours is left empty rather than summed short, because an
undercounted storm is worse than no number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_H = 24

# variable -> model-side inputs it is computed from
FORECAST_SOURCES = {
    "swe_24h_mm": ("precip_1h_mm",),
    "hn24_cm": ("precip_1h_mm", "air_temp_c"),
}
# variable -> observation column it is the 24 h gain of
OBS_SOURCES = {
    "swe_24h_mm": "swe_mm",
    "hn24_cm": "snow_depth_cm",
}
DERIVED = tuple(FORECAST_SOURCES)

# Issue-aligned rows (dynamical, and every live forecast) share an issue time within a
# forecast. Open-Meteo archive rows are continuous series per lead offset instead.
ISSUE_KEYED_SOURCES = {"dyn", "live"}


def is_derived(variable: str) -> bool:
    return variable in FORECAST_SOURCES


def fetch_variables(variables: list[str]) -> list[str]:
    """What must actually be fetched to produce ``variables``: derived ones replaced by
    their inputs, order kept, no duplicates."""
    out: list[str] = []
    for v in variables:
        for src in FORECAST_SOURCES.get(v, (v,)):
            if src not in out:
                out.append(src)
    return out


def snow_fraction(temp_c: np.ndarray) -> np.ndarray:
    """Share of precipitation falling as snow: all snow at or below 0 C, none at or above
    2 C, linear between. Near-surface temperature is a crude phase predictor, and the
    calibration's job is its site-specific error."""
    return np.clip((2.0 - temp_c) / 2.0, 0.0, 1.0)


def snow_to_liquid(temp_c: np.ndarray) -> np.ndarray:
    """Snow-to-liquid ratio from temperature: 10:1 near freezing, rising to 20:1 by
    -10 C and capped there. Colder snow is lighter. A simple stand-in for the
    Kuchera-style schemes, which need upper-air temperatures."""
    return np.clip(10.0 + np.maximum(-temp_c, 0.0), 10.0, 20.0)


def _series_keys(frame: pd.DataFrame) -> pd.Series:
    """Which rows belong to one continuous forecast series."""
    issue = frame["valid_time"] - pd.to_timedelta(frame["lead_h"], unit="h")
    keyed = frame["lead_source"].isin(ISSUE_KEYED_SOURCES)
    by_issue = frame["lead_source"].astype(str) + "|" + issue.astype(str)
    by_lead = frame["lead_source"].astype(str) + "|lead" + frame["lead_h"].astype(str)
    key = by_issue.where(keyed, by_lead)
    parts = [frame["model"].astype(str), key]
    if "station_id" in frame:
        parts.insert(0, frame["station_id"].astype(str))
    return pd.Series(["|".join(t) for t in zip(*parts)], index=frame.index)


def _window_sum(times: pd.Series, amounts: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Sum of ``amounts`` over the 24 h ending at each time, NaN where not fully covered."""
    s = pd.DataFrame({"a": amounts, "c": steps}, index=pd.DatetimeIndex(times))
    rolled = s.rolling(f"{WINDOW_H}h", closed="right", min_periods=1).sum()
    full = rolled["c"].to_numpy() >= WINDOW_H - 1e-6
    # A missing amount inside the window leaves its step uncounted, so coverage drops.
    return np.where(full, rolled["a"].to_numpy(), np.nan)


def add_forecast_columns(frame: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Add ``fc_{derived}`` columns to a long-form forecast frame.

    Each row's precipitation is a mean hourly rate over the step since the previous row
    of its series. Its amount is rate × step, and coverage is counted in hours. The
    first row of an issue-aligned series covers the hours since issue; the first row of
    an Open-Meteo archive series has no known step and so contributes no coverage.
    """
    wanted = [v for v in variables if is_derived(v)]
    if not wanted or frame.empty:
        return frame
    out = frame.copy()
    out["valid_time"] = pd.to_datetime(out["valid_time"])
    keys = _series_keys(out)
    for var in wanted:
        srcs = [f"fc_{s}" for s in FORECAST_SOURCES[var]]
        if not all(c in out for c in srcs):
            continue  # already derived upstream (a stored archive), or inputs absent
        out[f"fc_{var}"] = np.nan
    for _, idx in out.groupby(keys, sort=False).groups.items():
        block = out.loc[idx].sort_values("valid_time")
        steps = block["valid_time"].diff().dt.total_seconds().to_numpy() / 3600.0
        # An issue-aligned series starts at issue time, so its first row covers the
        # hours since issue. Without this, the window ending 24 h after issue ("new snow
        # by tomorrow morning", the one a briefing needs) was always one step short.
        if len(steps) and block["lead_source"].iloc[0] in ISSUE_KEYED_SOURCES:
            steps[0] = float(block["lead_h"].iloc[0])
        rate = block["fc_precip_1h_mm"].to_numpy(dtype=float) if "fc_precip_1h_mm" in block else None
        if rate is None:
            continue
        ok = np.isfinite(rate) & np.isfinite(steps)
        cover = np.where(ok, steps, 0.0)
        water = np.where(ok, rate * np.nan_to_num(steps), 0.0)
        for var in wanted:
            col = f"fc_{var}"
            if col not in out or not out.loc[block.index, col].isna().all():
                continue
            if var == "swe_24h_mm":
                amt = water
            else:  # hn24_cm
                temp = block["fc_air_temp_c"].to_numpy(dtype=float)
                t_ok = np.isfinite(temp)
                snow_mm = water * snow_fraction(np.nan_to_num(temp)) * snow_to_liquid(np.nan_to_num(temp))
                amt = np.where(t_ok, snow_mm / 10.0, 0.0)
                cover_v = np.where(t_ok, cover, 0.0)
                out.loc[block.index, col] = _window_sum(block["valid_time"], amt, cover_v)
                continue
            out.loc[block.index, col] = _window_sum(block["valid_time"], amt, cover)
    return out


# One reporting step of each SNOTEL sensor: SWE in 0.1 in, depth in whole inches. A 24 h
# gain no larger than one step is indistinguishable from the sensor flickering between
# two adjacent readings, which it does all summer over bare ground. Measured at
# Snowbird: with no snowpack, 4-21% of hours showed a SWE "gain" and 17-25% showed
# new snow before this floor. A real one-step gain is lost with it; the instrument
# cannot resolve it anyway.
NOISE_FLOOR = {"swe_24h_mm": 2.6, "hn24_cm": 2.6}
# The depth sensor also flickers by two steps (5.08 cm) over bare ground. A depth gain
# up to this size counts as new snow only when something else saw precipitation in
# the same 24 h: a SWE gain or measured precipitation. Larger gains stand alone.
HN24_CORROBORATE_BELOW_CM = 5.2
# Faster than any snowfall rate (about 10 in/h): a larger one-hour depth rise is a
# sensor step, and the 24 h windows containing it are unknown, not a storm.
MAX_HOURLY_SNOW_CM = 25.0
# A window that never went below this, with nothing else seeing precipitation, did not
# accumulate snow.
WARM_WINDOW_C = 5.0


def smooth_depth(depth: pd.Series) -> pd.Series:
    """Snow depth with spikes removed and sensor jitter damped.

    A reading more than 15 cm from the median of its surrounding 5 hours is a spike
    (a bird, a blowing-snow echo) and is dropped. A centred 3-hour median then removes
    the centimetre-scale jitter.
    """
    med5 = depth.rolling(5, center=True, min_periods=3).median()
    clean = depth.where((depth - med5).abs() <= 15.0)
    return clean.rolling(3, center=True, min_periods=2).median()


def _corroborate(gain: pd.Series, block: pd.DataFrame, hourly_index,
                 smoothed: pd.Series | None = None) -> pd.Series:
    """Zero small depth gains that no other sensor supports (see HN24_CORROBORATE_BELOW_CM)."""
    wet = pd.Series(False, index=hourly_index)
    by_time = block.set_index("valid_time")
    by_time = by_time[~by_time.index.duplicated(keep="last")].reindex(hourly_index)
    if "swe_mm" in by_time and by_time["swe_mm"].notna().any():
        swe = smooth_depth(by_time["swe_mm"].astype(float))
        wet |= ((swe - swe.shift(WINDOW_H)) > NOISE_FLOOR["swe_24h_mm"]).fillna(False)
    if "precip_1h_mm" in by_time and by_time["precip_1h_mm"].notna().any():
        p24 = by_time["precip_1h_mm"].astype(float).rolling(WINDOW_H, min_periods=1).sum()
        wet |= (p24 > 0.0).fillna(False)
    small = (gain > 0) & (gain < HN24_CORROBORATE_BELOW_CM)
    gain = gain.where(~(small & ~wet), 0.0)

    # Sensor steps. Lone Mountain's depth jumped 0 -> 129 cm in one hour on 25 June 2024,
    # at 20 C with no precipitation and no SWE change, and stayed there: a persistent
    # step survives median smoothing and read as a 51-inch storm. No snowfall rate
    # reaches 25 cm in an hour, and snow does not accumulate across a warm window
    # nothing else saw fall.
    # Checked on the smoothed depth: a one-hour spike is already gone there, and what
    # remains is a step that persists.
    depth = smoothed
    if depth is not None:
        jump = depth.ffill().diff().rolling(WINDOW_H, min_periods=1).max()
        stepped = (jump > MAX_HOURLY_SNOW_CM).fillna(False)
        gain = gain.where(~stepped, np.nan)
    if "air_temp_c" in by_time and by_time["air_temp_c"].notna().any():
        warm = (by_time["air_temp_c"].astype(float).rolling(WINDOW_H, min_periods=1).min()
                > WARM_WINDOW_C).fillna(False)
        gain = gain.where(~(warm & ~wet & (gain > 0)), 0.0)
    return gain


def add_obs_columns(obs: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Add derived observation columns: the 24 h gain in SWE or smoothed depth.

    Gains are clipped at zero: settlement and melt make depth fall, and what the
    variable measures is new snow, not net change. Settlement during a storm therefore
    biases observed HN24 low. That is a known property of depth-sensor HN24, and
    calibration learns the station's version of it.
    """
    wanted = [v for v in variables if is_derived(v)]
    if not wanted or obs.empty:
        return obs
    out = obs.copy()
    out["valid_time"] = pd.to_datetime(out["valid_time"])
    for var in wanted:
        out[var] = np.nan
    for _, idx in out.groupby("station_id", sort=False).groups.items():
        block = out.loc[idx].sort_values("valid_time")
        hourly_index = pd.date_range(block["valid_time"].min(), block["valid_time"].max(), freq="h")
        for var in wanted:
            col = OBS_SOURCES[var]
            if col not in block or block[col].isna().all():
                continue
            series = block.set_index("valid_time")[col].astype(float)
            series = series[~series.index.duplicated(keep="last")].reindex(hourly_index)
            series = smooth_depth(series)  # spikes and jitter, for SWE as for depth
            gain = (series - series.shift(WINDOW_H)).clip(lower=0.0)
            gain = gain.where(gain > NOISE_FLOOR[var], 0.0).where(gain.notna())
            if var == "hn24_cm":
                gain = _corroborate(gain, block, hourly_index, series)
            out.loc[block.index, var] = gain.reindex(block["valid_time"]).to_numpy()
    return out
