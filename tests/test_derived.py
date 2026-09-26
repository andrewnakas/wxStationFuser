"""24-hour snow and water amounts: windows, phase and density, depth smoothing, pairing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wxfuser.config import known_variables, model_supports
from wxfuser.data import derived, pairs
from wxfuser.data.obs import OBS_COLUMNS

ISSUE = pd.Timestamp("2026-01-10 00:00")


def _issue_rows(step_h=1, n=30, rate=1.0, temp=-10.0, model="hrrr", source="dyn"):
    leads = np.arange(1, n + 1) * step_h
    return pd.DataFrame({
        "model": model,
        "valid_time": ISSUE + pd.to_timedelta(leads, unit="h"),
        "lead_h": leads,
        "lead_source": source,
        "fc_precip_1h_mm": rate,
        "fc_air_temp_c": temp,
    })


def test_derived_variables_fetch_their_inputs():
    assert derived.fetch_variables(["hn24_cm", "air_temp_c", "swe_24h_mm"]) == [
        "precip_1h_mm", "air_temp_c"]
    assert {"hn24_cm", "swe_24h_mm"} <= set(known_variables())
    assert model_supports("gfs16", "hn24_cm")  # has precipitation and temperature
    assert not model_supports("nonexistent", "hn24_cm")


def test_window_sums_rate_times_step_and_refuses_partial_windows():
    rows = _issue_rows(n=40)
    rows.loc[rows["lead_h"] == 30, "fc_precip_1h_mm"] = np.nan  # one missing hour
    out = derived.add_forecast_columns(rows, ["swe_24h_mm"]).set_index("lead_h")
    assert out.loc[29, "fc_swe_24h_mm"] == pytest.approx(24.0)
    # Every window containing the gap is refused rather than summed short.
    assert out.loc[30:40, "fc_swe_24h_mm"].isna().all()


def test_three_hourly_steps_weigh_by_their_length():
    out = derived.add_forecast_columns(_issue_rows(step_h=3, n=12), ["swe_24h_mm"])
    assert out["fc_swe_24h_mm"].dropna().iloc[0] == pytest.approx(24.0)


def test_new_snow_depends_on_temperature():
    cold = derived.add_forecast_columns(_issue_rows(temp=-10.0), ["hn24_cm"])
    warm = derived.add_forecast_columns(_issue_rows(temp=5.0), ["hn24_cm"])
    # 24 mm of water at -10 C: all snow, 20:1, so 480 mm = 48 cm.
    assert cold["fc_hn24_cm"].dropna().iloc[0] == pytest.approx(48.0)
    assert warm["fc_hn24_cm"].dropna().iloc[0] == pytest.approx(0.0)


def test_separate_issues_never_share_a_window():
    a = _issue_rows(n=20)
    b = _issue_rows(n=20).assign(valid_time=lambda d: d["valid_time"] + pd.Timedelta(hours=6))
    out = derived.add_forecast_columns(pd.concat([a, b], ignore_index=True), ["swe_24h_mm"])
    # Neither 20-hour forecast covers a full day, and merged they would appear to.
    assert out["fc_swe_24h_mm"].isna().all()


def test_open_meteo_archive_rows_window_along_their_lead_series():
    rows = _issue_rows(n=30, source="prev_runs")
    rows["lead_h"] = 24  # a previous_day1 series: many valid times, one nominal lead
    out = derived.add_forecast_columns(rows, ["swe_24h_mm"])
    assert out["fc_swe_24h_mm"].dropna().iloc[0] == pytest.approx(24.0)


def _obs(depth, swe=None, start="2026-01-10"):
    n = len(depth)
    df = pd.DataFrame({"station_id": "766:UT:SNTL",
                       "valid_time": pd.date_range(start, periods=n, freq="h"),
                       "source": "SNOTEL"})
    for c in OBS_COLUMNS:
        if c not in df:
            df[c] = np.nan
    df["snow_depth_cm"] = depth
    df["swe_mm"] = swe if swe is not None else np.nan
    return df


def test_observed_new_snow_is_the_smoothed_depth_gain():
    depth = 100.0 + np.arange(48.0)  # 1 cm/h
    depth[30] = 190.0  # a spike
    out = derived.add_obs_columns(_obs(depth), ["hn24_cm"]).set_index("valid_time")
    assert out["hn24_cm"].iloc[40] == pytest.approx(24.0, abs=1.0)
    assert out["hn24_cm"].iloc[:24].isna().all()


def test_settling_snowpack_reports_no_new_snow():
    out = derived.add_obs_columns(_obs(200.0 - np.arange(48.0) * 0.5), ["hn24_cm"])
    assert (out["hn24_cm"].dropna() == 0).all()


def test_pairs_carry_derived_columns_end_to_end():
    fc = _issue_rows(n=40)
    ob = _obs(100.0 + np.arange(72.0), swe=np.arange(72.0) * 2.0, start="2026-01-09")
    built = pairs.build_pairs(fc, ob, "766:UT:SNTL", ["hn24_cm", "swe_24h_mm"])
    assert {"fc_hn24_cm", "obs_hn24_cm", "fc_swe_24h_mm", "obs_swe_24h_mm"} <= set(built.columns)
    row = built.dropna(subset=["fc_swe_24h_mm", "obs_swe_24h_mm"]).iloc[0]
    assert row["fc_swe_24h_mm"] == pytest.approx(24.0)
    assert row["obs_swe_24h_mm"] == pytest.approx(48.0)


def test_zero_inflated_amounts_use_the_two_part_model():
    from wxfuser.models import tier1_emos

    assert tier1_emos.distribution_for("hn24_cm") == "bernoulli_quantile_map"
    rng = np.random.default_rng(0)
    n = 400
    f = np.where(rng.random(n) < 0.3, rng.gamma(2.0, 5.0, n), 0.0)
    wide = pd.DataFrame({
        "valid_time": pd.date_range("2026-01-01", periods=n, freq="h"),
        "lead_h": 12, "lead_source": "dyn",
        "fc_hrrr": f, "obs": np.maximum(f * 1.3 + rng.normal(0, 1, n), 0) * (f > 0),
    })
    wide["fc_mean"], wide["fc_spread"] = wide["fc_hrrr"], 0.0
    state = tier1_emos.fit(wide, "hn24_cm", ["hrrr"])
    assert state["variable"] == "hn24_cm" and state["buckets"]
    pred = tier1_emos.predict(state, wide.iloc[:5], ["hrrr"])
    assert "p_occ" in pred
    assert all((pred[k] >= 0).all() for k in pred if k.startswith("q"))


def test_the_window_ending_24h_after_issue_is_complete():
    """'New snow by tomorrow morning' is the window a briefing needs; it must not be empty."""
    out = derived.add_forecast_columns(_issue_rows(), ["swe_24h_mm"]).set_index("lead_h")
    assert out.loc[24, "fc_swe_24h_mm"] == pytest.approx(24.0)
    assert np.isnan(out.loc[23, "fc_swe_24h_mm"])  # reaches back before issue
