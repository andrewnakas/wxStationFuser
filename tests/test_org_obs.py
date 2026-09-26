"""Org station observations: CSV ingest, push inbox, and aggregation on read."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wxfuser.data import obs, org_obs

SID = "ORG:wasatch-patrol:collins"


def _csv(tmp_path, start="2026-01-11 00:00", periods=18, name="log.csv", **extra):
    """A 10-minute logger file in Mountain time, Fahrenheit, mph and cumulative inches."""
    t = pd.date_range(start, periods=periods, freq="10min")
    df = pd.DataFrame({
        "Timestamp": t.strftime("%Y-%m-%d %H:%M"),
        "TempF": 32.0 + np.arange(periods),          # 0 C at the first reading
        "Gust": 10.0 + np.arange(periods) % 6,        # mph
        "PrecipIn": 1.0 + 0.01 * np.arange(periods),  # 0.01 in per reading
    })
    for k, v in extra.items():
        df[k] = v
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


MAPPING = org_obs.parse_mapping(
    ["air_temp_c=TempF:F", "wind_gust_ms=Gust:mph", "precip_mm=PrecipIn:in:cumulative"]
)


def test_csv_converts_units_zone_and_cumulative_precipitation(tmp_path):
    rec = org_obs.read_csv(_csv(tmp_path), SID, MAPPING, time_col="Timestamp",
                           tz="America/Denver")
    assert rec["time"].iloc[0] == pd.Timestamp("2026-01-11 07:00")  # MST = UTC-7
    assert rec["air_temp_c"].iloc[0] == pytest.approx(0.0)
    assert rec["wind_gust_ms"].iloc[0] == pytest.approx(10 * 0.44704)
    assert np.isnan(rec["precip_mm"].iloc[0])  # no increment before the first reading
    assert rec["precip_mm"].iloc[1] == pytest.approx(0.254)


def test_hourly_aggregation_labels_by_hour_end_and_sums_precip(tmp_path):
    root = tmp_path / "obs"
    rec = org_obs.read_csv(_csv(tmp_path), SID, MAPPING, time_col="Timestamp", tz="UTC")
    org_obs.append_raw(rec, root)
    hourly = org_obs.fetch_org_hourly(SID, "2026-01-11", "2026-01-11", root).set_index("valid_time")
    # 00:10..01:00 are the hour ending 01:00: six readings, five precip increments of
    # 0.254 mm plus the 00:10 increment, so six in all.
    h1 = hourly.loc[pd.Timestamp("2026-01-11 01:00")]
    assert h1["precip_1h_mm"] == pytest.approx(6 * 0.254)
    assert h1["wind_gust_ms"] == pytest.approx(15 * 0.44704)
    assert h1["air_temp_c"] == pytest.approx(((32 + np.arange(1, 7)) - 32).mean() * 5 / 9)


def test_reingesting_and_split_hours_aggregate_identically(tmp_path):
    whole, split = tmp_path / "whole", tmp_path / "split"
    rec = org_obs.read_csv(_csv(tmp_path), SID, MAPPING, time_col="Timestamp", tz="UTC")
    org_obs.append_raw(rec, whole)
    org_obs.append_raw(rec, whole)  # the same file twice
    org_obs.append_raw(rec.iloc[:4], split)  # an hour arriving in two pieces
    org_obs.append_raw(rec.iloc[4:], split)
    a = org_obs.fetch_org_hourly(SID, "2026-01-11", "2026-01-11", whole)
    b = org_obs.fetch_org_hourly(SID, "2026-01-11", "2026-01-11", split)
    pd.testing.assert_frame_equal(a, b)


def test_fixed_offset_loggers_on_standard_time(tmp_path):
    rec = org_obs.read_csv(_csv(tmp_path, start="2026-07-01 12:00"), SID, MAPPING,
                           time_col="Timestamp", tz="-07:00")
    # A logger on MST all year: noon in July is 19 UTC, not the 18 UTC daylight time gives.
    assert rec["time"].iloc[0] == pd.Timestamp("2026-07-01 19:00")


def test_diurnal_check_flags_local_time_read_as_utc(tmp_path):
    t = pd.date_range("2026-01-01", periods=24 * 7, freq="h")
    local_hour = (t.hour - 7) % 24  # true Mountain local hour
    temp = -5 + 5 * np.cos((local_hour - 15) / 24 * 2 * np.pi)  # peaks 15:00 local
    good = pd.DataFrame({"time": t, "air_temp_c": temp})
    assert 13 <= org_obs.diurnal_peak_solar_hour(good, -111.6) <= 16
    shifted = good.assign(time=good["time"] - pd.Timedelta(hours=7))  # local read as UTC
    assert not 11 <= org_obs.diurnal_peak_solar_hour(shifted, -111.6) <= 18


def test_bad_mappings_are_refused():
    with pytest.raises(org_obs.IngestError):
        org_obs.parse_mapping(["air_temp_c=Temp:mph"])
    with pytest.raises(org_obs.IngestError):
        org_obs.parse_mapping(["wind_gust_ms=Gust:mph:cumulative"])
    with pytest.raises(org_obs.IngestError):
        org_obs.parse_mapping(["snow=Depth:cm"])


def test_inbox_takes_only_this_orgs_stations(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    lines = [
        {"station": SID, "time": "2026-01-11T07:10:00-07:00", "air_temp_c": -6.0},
        {"station": "ORG:someone-else:x", "time": "2026-01-11T14:10:00Z", "air_temp_c": 1.0},
        "not json",
    ]
    (inbox / "2026-01-11.jsonl").write_text(
        "\n".join(json.dumps(x) if isinstance(x, dict) else x for x in lines)
    )
    held = org_obs.compact_inbox(inbox, "wasatch-patrol", tmp_path / "obs")
    assert held == {SID: 1}
    raw = pd.read_parquet(org_obs.store_path(SID, tmp_path / "obs"))
    assert raw["time"].iloc[0] == pd.Timestamp("2026-01-11 14:10")


def test_fetch_obs_routes_org_stations_to_their_store(tmp_path, monkeypatch):
    monkeypatch.setattr(org_obs, "ORG_OBS_DIR", tmp_path)
    rec = org_obs.read_csv(_csv(tmp_path), SID, MAPPING, time_col="Timestamp", tz="UTC")
    org_obs.append_raw(rec)
    got = obs.fetch_obs(SID, pd.Timestamp("2026-01-11").date(), pd.Timestamp("2026-01-11").date())
    assert list(got.columns) == obs.OBS_COLUMNS
    assert (got["source"] == "ORG").all()


def test_snow_depth_uploads_are_kept_in_centimetres(tmp_path):
    path = _csv(tmp_path, Depth=np.linspace(40.0, 41.0, 18))  # inches
    mapping = org_obs.parse_mapping(["snow_depth_cm=Depth:in"])
    rec = org_obs.read_csv(path, SID, mapping, time_col="Timestamp", tz="UTC")
    assert rec["snow_depth_cm"].iloc[0] == pytest.approx(101.6)
    org_obs.append_raw(rec, tmp_path / "obs")
    hourly = org_obs.fetch_org_hourly(SID, "2026-01-11", "2026-01-11", tmp_path / "obs")
    # A state, not an amount: the hour takes its last reading, never a sum.
    assert hourly["snow_depth_cm"].max() < 110
