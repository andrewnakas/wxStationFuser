"""The short-lead nowcast feature: the fused model's error just before issue."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wxfuser.data import pairs


def _pairs(n_issues=3, source="dyn"):
    rows = []
    for k in range(n_issues):
        issue = pd.Timestamp("2026-01-10") + pd.Timedelta(hours=6 * k)
        for lead in range(1, 13):
            vt = issue + pd.Timedelta(hours=lead)
            rows.append({"station_id": "S", "model": "hrrr", "valid_time": vt, "lead_h": lead,
                         "lead_source": source, "fc_air_temp_c": 0.0,
                         # obs runs 3 C warmer than the model, every hour
                         "obs_air_temp_c": 3.0, "obs_source": "x"})
    return pd.DataFrame(rows)


def test_nowcast_error_is_obs_an_hour_before_issue_minus_the_first_forecast_hour():
    wide = pairs.to_wide(_pairs(), "air_temp_c", ["hrrr"])
    later = wide[wide["valid_time"] >= pd.Timestamp("2026-01-10 06:00")]
    # From the second issue on, the hour before issue was observed (by the first issue's
    # rows), and the error it shows is the model's 3-degree cold bias.
    assert later["nowcast_err"].dropna().eq(3.0).all()
    assert later["nowcast_err"].notna().any()
    # The first issue has no observation before it.
    assert wide[wide["valid_time"] <= pd.Timestamp("2026-01-10 06:00")]["nowcast_err"].isna().all()


def test_open_meteo_archive_rows_have_no_issue_and_no_nowcast():
    wide = pairs.to_wide(_pairs(source="prev_runs"), "air_temp_c", ["hrrr"])
    assert wide["nowcast_err"].isna().all()


def test_tier3_takes_the_feature_only_when_training_data_carries_it():
    pytest.importorskip("lightgbm")
    from wxfuser.models import tier3_gbm

    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({
        "valid_time": pd.date_range("2025-01-01", periods=n, freq="h"),
        "lead_h": rng.integers(1, 48, n), "fc_hrrr": rng.normal(0, 5, n),
    })
    err = rng.normal(0, 2, n)
    df["obs"] = df["fc_hrrr"] + err * np.exp(-df["lead_h"] / 12)
    df["nowcast_err"] = err
    state = tier3_gbm.fit(df, "air_temp_c", ["hrrr"])
    assert "nowcast_err" in state["features"]
    without = tier3_gbm.fit(df.drop(columns="nowcast_err"), "air_temp_c", ["hrrr"])
    assert "nowcast_err" not in without["features"]
    # A model trained without the feature still predicts from a frame that has it.
    assert tier3_gbm.predict(without, df.iloc[:5], ["hrrr"])["calibrated"]
