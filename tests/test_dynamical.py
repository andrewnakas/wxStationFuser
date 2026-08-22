"""Reading forecasts out of the gridded archives.

Everything here runs against a fake store, because the properties worth pinning are the
ones that would corrupt a fit silently rather than fail: a unit left unconverted, a lead
offset applied to the wrong initialisation, a variable a model does not carry arriving as
a column of nulls, and — the one the whole design rests on — stations sharing a chunk
being served by a single read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wxfuser.data import dynamical

INITS = pd.date_range("2026-01-01", periods=4, freq="6h")
LEADS = np.arange(0, 12, 1.0)          # hourly, 0..11 h
NLAT, NLON = 8, 8
CHUNKS = (1, len(LEADS), 4, 4)          # four 4x4 tiles


class _OIndex:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return self.values[key]


class FakeArray:
    def __init__(self, values, chunks):
        self.values = np.asarray(values, dtype="float64")
        self.chunks = chunks
        self.reads = 0

    def __getitem__(self, key):
        return self.values[key]

    @property
    def oindex(self):
        self.reads += 1
        return _OIndex(self.values)


class FakeStore:
    """Just enough of a zarr group: named arrays, chunk shapes, and orthogonal indexing."""

    def __init__(self, names, *, lat=None, lon=None):
        self._arrays = {}
        base = np.arange(len(INITS) * len(LEADS) * NLAT * NLON, dtype="float64")
        base = base.reshape(len(INITS), len(LEADS), NLAT, NLON)
        for i, name in enumerate(names):
            self._arrays[name] = FakeArray(base + i * 1000.0, CHUNKS)
        # Seconds since the epoch, which is what the real stores hold.
        epoch = INITS.astype("datetime64[s]").astype("int64")
        self._arrays["init_time"] = FakeArray(epoch, (len(INITS),))
        self._arrays["lead_time"] = FakeArray(LEADS * 3600.0, (len(LEADS),))
        self._arrays["latitude"] = FakeArray(
            lat if lat is not None else np.linspace(50, 43, NLAT), (NLAT,)
        )
        self._arrays["longitude"] = FakeArray(
            lon if lon is not None else np.linspace(-110, -103, NLON), (NLON,)
        )

    def arrays(self):
        return list(self._arrays.items())

    def __getitem__(self, name):
        return self._arrays[name]


GLOBAL_NAMES = ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m",
                "precipitation_surface"]
FULL_NAMES = [*GLOBAL_NAMES, "wind_gust_surface"]
DEWPOINT_NAMES = ["temperature_2m", "dew_point_temperature_2m", "wind_u_10m", "wind_v_10m",
                  "precipitation_surface"]


@pytest.fixture(autouse=True)
def _no_caches():
    """The module memoises stores and grids; a test must not inherit another's."""
    for fn in (dynamical.open_store, dynamical._grid, dynamical._array_names,
               dynamical._tile_shape, dynamical._recipe_for, dynamical._projected_tree):
        fn.cache_clear()
    yield


def _install(monkeypatch, store):
    monkeypatch.setattr(dynamical, "open_store", lambda model: store)
    return store


def test_a_variable_the_archive_lacks_is_reported_missing_not_null(monkeypatch):
    """A null column is indistinguishable from a missing observation once it reaches a
    fit, so the absence has to be visible before the fetch, not after."""
    _install(monkeypatch, FakeStore(GLOBAL_NAMES))
    assert dynamical.carries("dyn_gfs", "air_temp_c")
    assert not dynamical.carries("dyn_gfs", "wind_gust_ms")


def test_gusts_are_found_under_either_name(monkeypatch):
    """HRRR calls them wind_gust_surface and ICON-EU wind_gust_10m."""
    _install(monkeypatch, FakeStore([*GLOBAL_NAMES, "wind_gust_10m"]))
    assert dynamical.carries("dyn_icon_eu", "wind_gust_ms")


def test_humidity_falls_back_to_dew_point(monkeypatch):
    """AIFS publishes no relative humidity at all; the alternative is nothing."""
    _install(monkeypatch, FakeStore(DEWPOINT_NAMES))
    assert dynamical.carries("dyn_aifs", "rh_pct")
    assert dynamical._recipe_for("dyn_aifs", "rh_pct") == (
        "temperature_2m", "dew_point_temperature_2m"
    )


def test_derived_humidity_is_saturated_when_dew_point_meets_temperature():
    rh = dynamical._relative_humidity(np.array([15.0, 15.0]), np.array([15.0, 5.0]))
    assert rh[0] == pytest.approx(100.0, abs=0.01)
    assert 45.0 < rh[1] < 55.0
    assert (rh <= 100.0).all()


def _fetch(monkeypatch, store, variables, **kw):
    _install(monkeypatch, store)
    return dynamical.fetch_history_batch(
        [("S1", 46.0, -106.0)], "dyn_gfs", variables,
        INITS[0].date(), (INITS[-1] + pd.Timedelta(days=1)).date(),
        init_stride_h=6, max_lead_h=11, lead_stride_h=1, **kw,
    )


def test_valid_time_is_the_initialisation_plus_the_lead(monkeypatch):
    out = _fetch(monkeypatch, FakeStore(GLOBAL_NAMES), ["air_temp_c"])
    assert not out.empty
    expected = pd.to_datetime(out["valid_time"]) - pd.to_timedelta(out["lead_h"], unit="h")
    # Every row must trace back to one of the four initialisations, exactly.
    assert set(expected.unique()) == set(INITS)
    assert out["lead_h"].min() == 0 and out["lead_h"].max() == 11


def test_wind_speed_is_the_magnitude_of_the_components(monkeypatch):
    store = FakeStore(GLOBAL_NAMES)
    store["wind_u_10m"].values[:] = 3.0
    store["wind_v_10m"].values[:] = 4.0
    out = _fetch(monkeypatch, store, ["wind_speed_ms"])
    assert out["fc_wind_speed_ms"].round(6).eq(5.0).all()


def test_precipitation_is_converted_from_a_rate_to_millimetres(monkeypatch):
    """Stored as kg m-2 s-1. Published untouched it would understate rain 3,600-fold."""
    store = FakeStore(GLOBAL_NAMES)
    store["precipitation_surface"].values[:] = 0.001      # 1 mm in 1000 s
    out = _fetch(monkeypatch, store, ["precip_1h_mm"])
    assert out["fc_precip_1h_mm"].round(6).eq(3.6).all()


def test_a_lead_stride_thins_the_leads_it_was_asked_to(monkeypatch):
    _install(monkeypatch, FakeStore(GLOBAL_NAMES))
    out = dynamical.fetch_history_batch(
        [("S1", 46.0, -106.0)], "dyn_gfs", ["air_temp_c"],
        INITS[0].date(), (INITS[-1] + pd.Timedelta(days=1)).date(),
        init_stride_h=6, max_lead_h=11, lead_stride_h=3,
    )
    assert sorted(out["lead_h"].unique()) == [0, 3, 6, 9]


def test_stations_sharing_a_tile_cost_one_read(monkeypatch):
    """The reason this is affordable at all.

    Four stations inside one 4x4 chunk must be served by the same slab, not by four
    reads of it. Per-station reads would multiply a fleet backfill by the number of
    stations — which for the busiest GFS tile is 2,422 of them.
    """
    store = _install(monkeypatch, FakeStore(GLOBAL_NAMES))
    together = [("A", 49.0, -109.0), ("B", 49.0, -108.0), ("C", 48.0, -109.0),
                ("D", 48.0, -108.0)]
    out = dynamical.fetch_history_batch(
        together, "dyn_gfs", ["air_temp_c"],
        INITS[0].date(), (INITS[-1] + pd.Timedelta(days=1)).date(),
        init_stride_h=6, max_lead_h=11, lead_stride_h=1,
    )
    assert set(out["station_id"]) == {"A", "B", "C", "D"}
    assert store["temperature_2m"].reads == 1

    # ... and they are not all handed the same cell's numbers.
    per_station = out.groupby("station_id")["fc_air_temp_c"].first()
    assert per_station.nunique() == 4


def test_stations_in_different_tiles_are_read_separately(monkeypatch):
    store = _install(monkeypatch, FakeStore(GLOBAL_NAMES))
    apart = [("A", 49.0, -109.0), ("B", 44.0, -104.0)]
    dynamical.fetch_history_batch(
        apart, "dyn_gfs", ["air_temp_c"],
        INITS[0].date(), (INITS[-1] + pd.Timedelta(days=1)).date(),
        init_stride_h=6, max_lead_h=11, lead_stride_h=1,
    )
    assert store["temperature_2m"].reads == 2


def test_a_longitude_given_east_of_the_meridian_finds_the_same_cell(monkeypatch):
    """Registries are not consistent about 0..360 versus -180..180, and the wrong
    convention lands a station on the far side of the world rather than failing."""
    _install(monkeypatch, FakeStore(GLOBAL_NAMES))
    assert dynamical.nearest_cell("dyn_gfs", 46.0, -106.0) == dynamical.nearest_cell(
        "dyn_gfs", 46.0, 254.0
    )


def test_an_empty_window_returns_nothing_rather_than_raising(monkeypatch):
    _install(monkeypatch, FakeStore(GLOBAL_NAMES))
    out = dynamical.fetch_history_batch(
        [("S1", 46.0, -106.0)], "dyn_gfs", ["air_temp_c"],
        pd.Timestamp("2020-01-01").date(), pd.Timestamp("2020-02-01").date(),
    )
    assert out.empty


def test_the_frame_matches_what_the_pairing_code_expects(monkeypatch):
    """build_pairs reads these column names; a rename here breaks the archive silently."""
    from wxfuser.data import pairs as pairs_mod

    out = _fetch(monkeypatch, FakeStore(GLOBAL_NAMES), ["air_temp_c", "wind_speed_ms"])
    for column in ("station_id", "valid_time", "lead_h", "model", "lead_source"):
        assert column in out.columns

    obs = pd.DataFrame(
        {
            "station_id": "S1",
            "valid_time": pd.to_datetime(out["valid_time"]).unique(),
            "air_temp_c": 10.0,
            "source": "TEST",
        }
    )
    built = pairs_mod.build_pairs(out, obs, "S1", ["air_temp_c", "wind_speed_ms"])
    assert not built.empty
    assert {"fc_air_temp_c", "obs_air_temp_c", "lead_h"} <= set(built.columns)


def test_the_default_model_sets_never_mix_providers():
    """A mixed set does not fuse — see configs/models.yaml.

    `to_wide` keys on lead_source, so the two providers' rows never share a row and the
    fused forecast silently becomes a one-model one. Nothing raises; the site just quotes
    a spread that was never computed from more than one model.
    """
    from wxfuser.config import load_configs
    from wxfuser.pipeline.bulk_run import split_by_provider

    defaults = load_configs()["models"]["defaults"]
    for name, models in defaults.items():
        if not name.startswith("models"):
            continue
        point, grid = split_by_provider(list(models))
        assert not (point and grid), f"{name} mixes providers: point={point} grid={grid}"


def test_the_live_fetch_takes_only_the_newest_run(monkeypatch):
    """Four runs share the last day in the fixture; a live forecast is one of them.

    Blending several initialisations for the same valid hour publishes a mixture of runs
    dressed as the current one — and the older members of that mixture are, by
    construction, the worse forecasts.
    """
    _install(monkeypatch, FakeStore(GLOBAL_NAMES))
    out = dynamical.fetch_forecast_batch(
        [("S1", 46.0, -106.0)], ["dyn_gfs"], ["air_temp_c"], max_lead_h=11
    )
    assert not out.empty
    implied = pd.to_datetime(out["valid_time"]) - pd.to_timedelta(out["lead_h"], unit="h")
    assert set(implied.unique()) == {INITS[-1]}
