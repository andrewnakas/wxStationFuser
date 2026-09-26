"""dynamical.org adapter: grid lookup, unit handling, and issue-time alignment.

The stores are replaced by small in-memory datasets with the same layout (dims, names,
units) as the real ones, so these run offline.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from wxfuser.data import dynamical, pairs, sources  # noqa: E402


def _latlon_store(inits, leads_h, *, members=0, lon0=-115.0):
    """A GEFS/GFS-shaped store: (init_time, [ensemble_member,] lead_time, lat, lon)."""
    lat = np.arange(38.0, 42.01, 0.25)[::-1]  # descending, as dynamical stores it
    lon = np.arange(lon0, lon0 + 6.01, 0.25)
    shape = [len(inits), len(leads_h), len(lat), len(lon)]
    dims = ["init_time", "lead_time", "latitude", "longitude"]
    if members:
        shape.insert(1, members)
        dims.insert(1, "ensemble_member")
    lead_td = pd.to_timedelta(leads_h, unit="h")

    base = np.zeros(shape)
    # Temperature encodes the init index and lead so a test can tell which run it read.
    ti = np.arange(len(inits)).reshape([-1] + [1] * (len(shape) - 1))
    li_axis = dims.index("lead_time")
    li = np.asarray(leads_h, dtype=float).reshape([-1 if i == li_axis else 1 for i in range(len(shape))])
    temp = base + 100.0 * ti + li
    if members:
        m = np.arange(members).reshape([1, -1] + [1] * (len(shape) - 2))
        temp = temp + (m - (members - 1) / 2.0)  # members symmetric about the mean
    data = {
        "temperature_2m": (dims, temp),
        "dew_point_temperature_2m": (dims, temp - 5.0),
        "relative_humidity_2m": (dims, base + 55.0),
        "wind_u_10m": (dims, base + 3.0),
        "wind_v_10m": (dims, base + 4.0),
        "wind_gust_surface": (dims, base + 9.0),
        "precipitation_surface": (dims, base + 1.0 / 3600.0),  # 1 mm/h as kg m-2 s-1
    }
    coords = {"init_time": pd.to_datetime(inits), "lead_time": lead_td,
              "latitude": lat, "longitude": lon}
    if members:
        coords["ensemble_member"] = np.arange(members)
    return xr.Dataset(data, coords=coords)


def _projected_store(inits, leads_h):
    """An HRRR-shaped store: 2-D latitude/longitude on y/x."""
    y, x = np.arange(20), np.arange(30)
    lat2 = 38.0 + 0.1 * y[:, None] + 0.01 * x[None, :]
    lon2 = -115.0 + 0.1 * x[None, :] + 0.0 * y[:, None]
    shape = (len(inits), len(leads_h), len(y), len(x))
    dims = ["init_time", "lead_time", "y", "x"]
    temp = np.zeros(shape) + np.arange(len(inits)).reshape(-1, 1, 1, 1) * 100.0
    temp = temp + np.asarray(leads_h, dtype=float).reshape(1, -1, 1, 1)
    return xr.Dataset(
        {"temperature_2m": (dims, temp)},
        coords={
            "init_time": pd.to_datetime(inits),
            "lead_time": pd.to_timedelta(leads_h, unit="h"),
            "latitude": (("y", "x"), lat2),
            "longitude": (("y", "x"), lon2),
        },
    )


STATION = [("SNOTEL:1", 40.6, -111.6)]


def test_latlon_lookup_wraps_longitude_conventions():
    ds = _latlon_store(["2025-01-01"], [0, 1], lon0=245.0)  # 0..360 longitudes
    idx = dynamical.grid_indices(ds, np.array([40.6]), np.array([-111.6]))
    assert ds["latitude"].values[idx["latitude"][0]] == pytest.approx(40.5)
    assert ds["longitude"].values[idx["longitude"][0]] == pytest.approx(248.5)


def test_projected_lookup_and_domain_check():
    ds = _projected_store(["2025-01-01"], [0, 1])
    lats, lons = np.array([39.0, 10.0]), np.array([-114.0, -114.0])
    idx = dynamical.grid_indices(ds, lats, lons)
    inside = dynamical._in_domain(ds, lats, lons, idx)
    assert inside.tolist() == [True, False]
    j, i = idx["y"][0], idx["x"][0]
    assert abs(ds["latitude"].values[j, i] - 39.0) < 0.1


def test_extract_runs_converts_units_and_averages_members():
    ds = _latlon_store(["2025-01-01T00"], [0, 3, 6], members=5)
    runs = dynamical.extract_runs(
        "gefs", STATION, [0],
        ["air_temp_c", "rh_pct", "wind_speed_ms", "wind_gust_ms", "precip_1h_mm"], ds=ds,
    )
    assert len(runs) == 3
    # Member mean recovers the unperturbed value (init 0 -> 0 + lead).
    assert runs["fc_air_temp_c"].tolist() == pytest.approx([0.0, 3.0, 6.0])
    assert runs["fc_wind_speed_ms"].iloc[0] == pytest.approx(5.0)  # hypot(3, 4)
    assert runs["fc_precip_1h_mm"].iloc[0] == pytest.approx(1.0)
    assert runs["fc_rh_pct"].iloc[0] == pytest.approx(55.0)


def test_rh_is_derived_from_dewpoint_where_the_store_has_no_rh():
    ds = _latlon_store(["2025-01-01T00"], [0, 6], members=3)
    runs = dynamical.extract_runs("ecmwf_ens", STATION, [0], ["rh_pct"], ds=ds)
    t, td = 6.0, 1.0
    expected = 100 * np.exp(17.625 * td / (243.04 + td) - 17.625 * t / (243.04 + t))
    assert runs["fc_rh_pct"].iloc[1] == pytest.approx(expected, rel=1e-6)


def test_a_variable_the_model_lacks_stays_missing():
    ds = _latlon_store(["2025-01-01T00"], [0, 1])
    runs = dynamical.extract_runs("gfs16", STATION, [0], ["air_temp_c", "wind_gust_ms"], ds=ds)
    assert runs["fc_wind_gust_ms"].isna().all()
    assert runs["fc_air_temp_c"].notna().all()


def test_issue_alignment_respects_latency():
    """An issue may only use a run that had arrived: init + latency <= issue."""
    inits = pd.date_range("2025-01-01T00", periods=4, freq="6h")
    ds = _latlon_store(inits, list(range(0, 49)))
    # HRRR latency is 2 h: the 06z issue must still use the 00z run, not the 06z one.
    out = dynamical.fetch_archive_batch(
        STATION, ["hrrr"], ["air_temp_c"], date(2025, 1, 1), date(2025, 1, 1),
        issue_hours=[6], open_fn=lambda m: ds,
    )
    assert not out.empty
    first = out.sort_values("lead_h").iloc[0]
    assert first["lead_h"] == 1
    assert first["valid_time"] == pd.Timestamp("2025-01-01T07")
    # Temperature is 100 * run index + run lead, so 7 means the 00z run at its 7 h lead.
    assert first["fc_air_temp_c"] == pytest.approx(7.0)
    assert (out["lead_source"] == "dyn").all()


def test_models_on_different_cycles_share_rows_after_alignment():
    """The point of issue alignment: HRRR (6-hourly) and GEFS (daily) land side by side."""
    hrrr = _latlon_store(pd.date_range("2025-01-01", periods=8, freq="6h"), list(range(0, 49)))
    gefs = _latlon_store(pd.date_range("2025-01-01", periods=2, freq="D"), list(range(0, 169, 3)),
                         members=3)
    stores = {"hrrr": hrrr, "gefs": gefs}
    fc = dynamical.fetch_archive_batch(
        STATION, ["hrrr", "gefs"], ["air_temp_c"], date(2025, 1, 1), date(2025, 1, 2),
        issue_hours=[12], open_fn=stores.__getitem__,
    ).drop(columns="station_id")
    obs = pd.DataFrame({
        "valid_time": pd.date_range("2025-01-01", periods=96, freq="h"),
        "source": "test",
        "air_temp_c": 1.0,
    })
    built = pairs.build_pairs(fc, obs, "SNOTEL:1", ["air_temp_c"])
    wide = pairs.to_wide(built, "air_temp_c", ["hrrr", "gefs"])
    both = wide.dropna(subset=["fc_hrrr", "fc_gefs"])
    assert len(both) > 0
    # GEFS is 3-hourly, so every shared row falls on a GEFS step inside HRRR's 48 h.
    assert both["lead_h"].max() <= 48


def test_stale_runs_do_not_serve_an_issue():
    """A gap in the store is a gap, not a longer lead from an old run."""
    ds = _latlon_store(["2025-01-01T00"], list(range(0, 49)))
    out = dynamical.fetch_archive_batch(
        STATION, ["hrrr"], ["air_temp_c"], date(2025, 1, 5), date(2025, 1, 5),
        issue_hours=[0], open_fn=lambda m: ds,
    )
    assert out.empty


def test_live_steps_back_when_the_newest_run_is_still_empty():
    inits = pd.date_range("2025-01-01T00", periods=2, freq="6h")
    ds = _latlon_store(inits, list(range(0, 49)))
    ds["temperature_2m"][dict(init_time=1)] = np.nan  # metadata landed, data has not
    live = dynamical.fetch_live_batch(
        STATION, ["hrrr"], ["air_temp_c"], now=pd.Timestamp("2025-01-01T09:30"),
        open_fn=lambda m: ds,
    )
    assert not live.empty
    assert live["lead_h"].min() == 1
    # Run 0 at lead 10 is valid at 10z, one hour after the 09z issue.
    first = live.sort_values("lead_h").iloc[0]
    assert first["fc_air_temp_c"] == pytest.approx(10.0)


def test_sources_refuse_a_mixed_model_list():
    assert sources.source_for(["hrrr", "gefs"]) == "dynamical"
    assert sources.source_for(["gfs_seamless"]) == "openmeteo"
    with pytest.raises(sources.MixedSourceError):
        sources.source_for(["hrrr", "gfs_seamless"])


def test_every_dynamical_model_names_its_tree60_counterpart():
    from wxfuser.config import model_catalogue

    for mid, meta in model_catalogue().items():
        if meta.get("source") != "dynamical":
            continue
        assert meta.get("tree60_id"), f"{mid} has no tree60_id"
        assert meta.get("url", "").startswith("https://data.dynamical.org/")
        for var in meta["vars"]:
            assert var in meta["fields"], f"{mid} lists {var} but cannot derive it"
