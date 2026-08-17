"""Bulk observation sources, and the timestamp convention everything depends on.

The paired archive joins observations to forecasts on ``valid_time``. Forecasts are
UTC-naive, so observations must be too. A source that returns local or timezone-aware
timestamps does not fail — it pairs each observation with the forecast for a different
hour, by that station's UTC offset. Training and verification then share the same offset,
so the scorecard looks healthy while every published forecast is hours out of phase.

That is exactly what the dynamical.org ASOS archive does by default: it stores `valid` as
a timestamptz, which DuckDB renders in the session timezone inherited from the host.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wxfuser.data import bulk
from wxfuser.data.obs import OBS_COLUMNS


def _raw(valid_values) -> pd.DataFrame:
    n = len(valid_values)
    return pd.DataFrame(
        {
            "station": ["DEN"] * n,
            "valid": valid_values,
            "tmpc": np.linspace(10, 12, n),
            "dwpc": np.linspace(2, 3, n),
            "relh": np.linspace(50, 60, n),
            "sknt": np.full(n, 10.0),
            "gust": np.full(n, 20.0),
            "drct": np.full(n, 270.0),
            "p01m": np.zeros(n),
        }
    )


def test_normalised_timestamps_are_utc_naive():
    """A tz-aware source must be converted, not carried through."""
    aware = pd.to_datetime(
        ["2025-08-01 00:53", "2025-08-01 01:53"]
    ).tz_localize("America/Denver")
    out = bulk._normalise_asos(_raw(aware))
    assert out["valid_time"].dt.tz is None, "timestamps must be tz-naive"
    # 00:53 Mountain is 06:00 UTC after flooring — not 00:00.
    assert out["valid_time"].iloc[0] == pd.Timestamp("2025-08-01 06:00:00")


def test_naive_timestamps_pass_through_unchanged():
    naive = pd.to_datetime(["2025-08-01 06:53", "2025-08-01 07:53"])
    out = bulk._normalise_asos(_raw(naive))
    assert out["valid_time"].iloc[0] == pd.Timestamp("2025-08-01 06:00:00")


def test_local_and_utc_inputs_agree_on_the_same_instant():
    """The same moment expressed two ways must normalise to one timestamp.

    This is the property that was broken: the archive's instant was always right, but its
    rendering depended on the machine running the query.
    """
    local = pd.to_datetime(["2025-08-01 00:53"]).tz_localize("America/Denver")
    utc = local.tz_convert("UTC").tz_localize(None)
    a = bulk._normalise_asos(_raw(local))["valid_time"].iloc[0]
    b = bulk._normalise_asos(_raw(utc))["valid_time"].iloc[0]
    assert a == b


def test_units_are_converted_to_the_common_schema():
    naive = pd.to_datetime(["2025-08-01 06:53"])
    out = bulk._normalise_asos(_raw(naive))
    # Knots in, metres per second out.
    assert out["wind_speed_ms"].iloc[0] == pytest.approx(10.0 * 0.514444, rel=1e-6)
    assert out["wind_gust_ms"].iloc[0] == pytest.approx(20.0 * 0.514444, rel=1e-6)
    # Temperature and precipitation are already metric in this archive.
    assert out["air_temp_c"].iloc[0] == pytest.approx(10.0)
    assert list(out.columns) == OBS_COLUMNS


def test_subhourly_observations_collapse_to_the_hour():
    """Several METARs in one hour must become one row, gusts taken as the maximum."""
    naive = pd.to_datetime(["2025-08-01 06:05", "2025-08-01 06:35", "2025-08-01 06:55"])
    raw = _raw(naive)
    raw["gust"] = [10.0, 30.0, 20.0]
    out = bulk._normalise_asos(raw)
    assert len(out) == 1
    assert out["wind_gust_ms"].iloc[0] == pytest.approx(30.0 * 0.514444, rel=1e-6)


def test_station_ids_carry_the_network_prefix():
    out = bulk._normalise_asos(_raw(pd.to_datetime(["2025-08-01 06:53"])))
    assert out["station_id"].iloc[0] == "ASOS:DEN"


def test_empty_station_list_returns_the_empty_schema():
    from datetime import date

    out = bulk.asos_observations([], date(2025, 1, 1), date(2025, 1, 2))
    assert out.empty
    assert list(out.columns) == OBS_COLUMNS


# --------------------------------------------------- routing to the batched fetchers

def test_snotel_triplets_group_under_snotel():
    """A SNOTEL id names its network last, so the leading field must not decide routing.

    ``1000:OR:SNTL`` begins with a bare number. Grouping on the leading colon files each
    site under its own id, leaves the SNOTEL group empty, and drops every one of them into
    the per-station fallback — one HTTP request each, for the network that has no bulk
    archive and most needs the batching.
    """
    from wxfuser.pipeline.bulk_run import _source_of

    assert _source_of("1000:OR:SNTL") == "SNOTEL"
    assert _source_of("663:CO:SNTL") == "SNOTEL"
    assert _source_of("1165:MT:SNTLT") == "SNOTEL"


def test_prefixed_networks_still_route_on_their_prefix():
    from wxfuser.pipeline.bulk_run import _source_of

    assert _source_of("ASOS:KDEN") == "ASOS"
    assert _source_of("MS:10637") == "MS"
    assert _source_of("IEM:DEN") == "IEM"


def test_every_registry_station_reaches_a_batched_fetcher():
    """No enrolled station should silently fall back to one-request-per-station."""
    from wxfuser.data.registry import load_registry
    from wxfuser.pipeline.bulk_run import _source_of

    batched = {"ASOS", "SNOTEL", "MS"}
    stragglers = [s.id for s in load_registry() if _source_of(s.id) not in batched]
    assert len(stragglers) <= 1, f"{len(stragglers)} stations on the slow path: {stragglers[:5]}"


# ------------------------------------------------- publishing through a lagging source

def _station():
    from wxfuser.data.registry import Station

    return Station(id="MS:10637", name="Test", lat=50.0, lon=8.0, elev_m=100.0)


def test_stale_source_still_publishes_from_stored_history(monkeypatch, tmp_path):
    """An empty refresh window must not discard a station that already has an archive.

    Meteostat's bulk archive trails by months, so its stations routinely return nothing for
    a ten-day window. Bailing on that dropped 4,000 stations from the site while their
    paired history sat on disk, perfectly usable for training.
    """
    from wxfuser.pipeline import bulk_run, core

    archive = pd.DataFrame({"valid_time": pd.to_datetime(["2026-03-29 12:00"])})
    monkeypatch.setattr(core, "STATE_DIR", tmp_path)
    monkeypatch.setattr(core, "update_archive", lambda st, built: archive)
    monkeypatch.setattr(core, "train_variable", lambda *a, **k: {"status": "no"})

    live = pd.DataFrame({"valid_time": pd.to_datetime(["2026-08-17 00:00"]), "lead_h": [1]})
    out = bulk_run._finish_station(
        _station(), None, None, live, ["air_temp_c"], evaluate=False
    )
    assert out["status"] == "warming_up", out


def test_a_station_with_neither_window_nor_archive_is_reported_honestly(monkeypatch, tmp_path):
    from wxfuser.pipeline import bulk_run, core

    monkeypatch.setattr(core, "STATE_DIR", tmp_path)
    monkeypatch.setattr(core, "update_archive", lambda st, built: pd.DataFrame())

    live = pd.DataFrame({"valid_time": pd.to_datetime(["2026-08-17 00:00"]), "lead_h": [1]})
    out = bulk_run._finish_station(
        _station(), None, None, live, ["air_temp_c"], evaluate=False
    )
    assert out["status"] == "no_observations"
