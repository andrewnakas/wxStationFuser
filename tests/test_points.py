"""Point archives: month selection, init-hour filtering, and resumable output."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from wxfuser import points  # noqa: E402


def _latlon_store(inits, leads_h):
    """A GFS-shaped store whose temperature is 100 x run index + lead."""
    lat = np.arange(38.0, 42.01, 0.25)[::-1]
    lon = np.arange(-115.0, -108.99, 0.25)
    shape = (len(inits), len(leads_h), len(lat), len(lon))
    temp = (np.arange(len(inits)).reshape(-1, 1, 1, 1) * 100.0
            + np.asarray(leads_h, dtype=float).reshape(1, -1, 1, 1) + np.zeros(shape))
    dims = ["init_time", "lead_time", "latitude", "longitude"]
    return xr.Dataset({"temperature_2m": (dims, temp)}, coords={
        "init_time": pd.to_datetime(inits), "lead_time": pd.to_timedelta(leads_h, unit="h"),
        "latitude": lat, "longitude": lon})


def test_months_and_init_hour_filter():
    assert points.months(date(2025, 11, 20), date(2026, 2, 1)) == [
        "2025-11", "2025-12", "2026-01", "2026-02"]
    ds = _latlon_store(pd.date_range("2025-12-31T00", periods=12, freq="6h"), [0, 1])
    assert points.month_positions(ds, "2026-01", [0, 12]) == [4, 6, 8, 10]
    assert len(points.month_positions(ds, "2026-01", None)) == 8


def test_extract_month_keeps_raw_runs_with_leads():
    ds = _latlon_store(pd.date_range("2026-01-01", periods=4, freq="6h"), list(range(0, 7)))
    out = points.extract_month("hrrr", [("S1", 40.5, -111.6)], "2026-01", init_hours=[0, 12], ds=ds)
    assert sorted(out["init_time"].unique()) == [pd.Timestamp("2026-01-01T00"), pd.Timestamp("2026-01-01T12")]
    assert out["lead_h"].max() == 6 and out["lead_h"].dtype == np.int16
    assert out["fc_air_temp_c"].dtype == np.float32
    row = out[(out["init_time"] == pd.Timestamp("2026-01-01T12")) & (out["lead_h"] == 3)].iloc[0]
    assert row["fc_air_temp_c"] == pytest.approx(203.0)  # run index 2 * 100 + lead 3
