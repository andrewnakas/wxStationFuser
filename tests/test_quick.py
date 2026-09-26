"""wxfuser calibrate: link recognition, CSV detection, time-zone choice, report."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wxfuser import quick
from wxfuser.report import render


@pytest.mark.parametrize("link,kind,ids", [
    ("https://wcc.sc.egov.usda.gov/nwcc/site?sitenum=766", "snotel", {"site": "766"}),
    ("766:UT:SNTL", "snotel", {"triplet": "766:UT:SNTL"}),
    ("https://nwcc-apps.sc.egov.usda.gov/imap/#station=766_UT_SNTL", "snotel", {"triplet": "766:UT:SNTL"}),
    ("https://mesonet.agron.iastate.edu/sites/site.php?station=SLC&network=UT_ASOS", "iem",
     {"station": "SLC", "network": "UT_ASOS"}),
    ("https://mesowest.utah.edu/cgi-bin/droman/meso_base_dyn.cgi?stn=ATH20", "synoptic", {"stid": "ATH20"}),
    ("https://explore.synopticdata.com/ATH20/", "synoptic", {"stid": "ATH20"}),
    ("https://docs.google.com/spreadsheets/d/abc123/edit#gid=42", "csv",
     {"url": "https://docs.google.com/spreadsheets/d/abc123/export?format=csv&gid=42"}),
    ("https://example.com/plot.csv", "csv", {"url": "https://example.com/plot.csv"}),
])
def test_links_are_recognised(link, kind, ids):
    assert quick.classify(link) == (kind, ids)


def _logger_csv(tmp_path, offset_h=-7):
    """A week of 15-minute readings on local standard time, US units, with a preamble."""
    t_utc = pd.date_range("2026-01-10", periods=7 * 96, freq="15min")
    local_solar = (t_utc.hour + t_utc.minute / 60 - 111.6 / 15) % 24
    temp_c = -8 + 6 * np.cos((local_solar - 14.5) / 24 * 2 * np.pi)
    t_local = t_utc + pd.Timedelta(hours=offset_h)
    df = pd.DataFrame({
        "TIMESTAMP": t_local.strftime("%Y-%m-%d %H:%M"),
        "AirTemp_F": temp_c * 9 / 5 + 32,
        "DewPoint_F": temp_c * 9 / 5 + 20,
        "WindSpd_mph": 10.0,
        "WindGust_mph": 22.0,
        "Precip_Accum_in": np.round(np.arange(len(t_utc)) * 0.0005 + 3.0, 3),
        "SnowDepth_in": 60.0,
    })
    path = tmp_path / "plot.csv"
    path.write_text("Campbell Scientific CR1000,Upper Plot\nunits,\n" + df.to_csv(index=False))
    return path


def test_logger_columns_units_and_zone_are_detected(tmp_path):
    path = _logger_csv(tmp_path)
    df = quick.load_csv(str(path))
    time_col, mapping, notes = quick.detect_columns(df)
    assert time_col == "TIMESTAMP"
    assert set(mapping) == {
        "air_temp_c=AirTemp_F:F", "wind_speed_ms=WindSpd_mph:mph", "wind_gust_ms=WindGust_mph:mph",
        "precip_mm=Precip_Accum_in:in:cumulative", "snow_depth_cm=SnowDepth_in:in"}
    assert not any("DewPoint" in m for m in mapping)  # dew point is never temperature
    from wxfuser.data import org_obs

    df.to_csv(tmp_path / "clean.csv", index=False)
    tz, why = quick.choose_time_zone(str(tmp_path / "clean.csv"), "ORG:adhoc:p",
                                     org_obs.parse_mapping(mapping), time_col, -111.6)
    assert tz == "-07:00"


def test_a_utc_logger_is_read_as_utc(tmp_path):
    path = _logger_csv(tmp_path, offset_h=0)
    df = quick.load_csv(str(path))
    time_col, mapping, _ = quick.detect_columns(df)
    from wxfuser.data import org_obs

    df.to_csv(tmp_path / "clean.csv", index=False)
    tz, _ = quick.choose_time_zone(str(tmp_path / "clean.csv"), "ORG:adhoc:p",
                                   org_obs.parse_mapping(mapping), time_col, -111.6)
    assert tz == "UTC"


def test_variables_follow_the_data_and_models_follow_the_region():
    assert quick.variables_for({"air_temp_c", "precip_mm", "snow_depth_cm"}) == [
        "air_temp_c", "precip_1h_mm", "hn24_cm"]
    assert quick.default_models(40.6, -111.6) == ["hrrr", "gefs"]
    assert quick.default_models(46.0, 7.5) == ["gefs", "ecmwf_ens"]


def test_csv_needs_a_location():
    with pytest.raises(quick.LinkError, match="lat"):
        quick.resolve("https://example.com/plot.csv")


def test_report_renders_standalone_and_honest():
    times = pd.date_range("2026-01-11", periods=6, freq="h").strftime("%Y-%m-%dT%H:%MZ").tolist()
    fc = {
        "generated_at": "2026-01-11T00:10:00+00:00", "obs_latest": "2026-01-10T23:00Z",
        "station": {"name": "Upper Plot", "elev_m": 2900.0}, "models_used": ["hrrr", "gefs"],
        "spec": {"label": "Upper Plot"}, "method": {"wind_gust_ms": {"label": "EMOS", "train_days": 90}},
        "hourly": {"time": times, "wind_gust_ms": {
            "q05": [5.0] * 6, "q25": [8.0] * 6, "q50": [10.0] * 6, "q75": [12.0] * 6, "q95": [20.0] * 6,
            "p_exceed": {"18": [0.004, 0.1, 0.2, 0.3, 0.2, 0.1]}}},
        "raw": {"hrrr": {"wind_gust_ms": [9.0] * 6}},
        "skill": {"wind_gust_ms": {"status": "verified", "crpss_vs_raw": 0.05,
                                   "crpss_vs_raw_ci90": [-0.02, 0.11], "beats_raw": 0,
                                   "raw_best_model": "hrrr", "coverage90": 0.9, "coverage50": 0.5}},
    }
    out = render(fc, now=pd.Timestamp("2026-01-11T00:30Z").to_pydatetime())
    assert "<script" not in out and "http" not in out.split("</style>")[0]
    assert "&gt; 40 mph" in out
    assert "not yet distinguishable" in out  # no claim without a clear interval
    assert "<svg" in out
    json.dumps(fc)  # the fixture itself is valid schema-shaped JSON
