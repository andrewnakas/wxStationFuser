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


# ------------------------------------------------ which airports count as enrollable

class _FakeCon:
    """A DuckDB stand-in that records the SQL instead of running it."""

    def __init__(self):
        self.sql = None

    def execute(self, q):
        self.sql = q
        return self

    def fetchdf(self):
        return pd.DataFrame(
            {"station": ["DEN"], "sname": ["Denver"], "lat": [39.8], "lon": [-104.7],
             "elev_m": [1656.0], "country": ["US"], "state": ["CO"], "n_obs": [1]}
        )


def _fake_archive(monkeypatch):
    con = _FakeCon()
    monkeypatch.setattr(bulk, "_duckdb", lambda: con)
    monkeypatch.setattr(bulk, "_existing_year_urls",
                        lambda a, b: [f"u{y}" for y in range(a, b + 1)])
    return con


def test_the_window_is_a_rolling_year_not_the_previous_calendar_one(monkeypatch):
    """The bug this replaces: 2026 stations were being judged on their 2025 record.

    Measured against the live archive, that hid 279 stations — Ship Shoal, Pipestone and
    Creede among them, all of which now report more often than hourly. A station
    commissioned in March was invisible until the following January.
    """
    from datetime import date

    con = _fake_archive(monkeypatch)
    bulk.asos_stations(end=date(2026, 8, 22), days=365)

    assert "'u2025'" in con.sql and "'u2026'" in con.sql   # both partitions
    assert "valid >= TIMESTAMP '2025-08-22" in con.sql     # and only the trailing year


def test_a_station_reporting_at_all_is_enrollable(monkeypatch):
    """Observations come in bulk, so the only reason to leave an airport out is that it
    is dead. The old bar excluded live stations to save a forecast budget that a few
    dozen sparse reporters barely move."""
    con = _fake_archive(monkeypatch)
    out = bulk.asos_stations()

    assert "count(*) >= 1" in con.sql
    assert list(out["id"]) == ["ASOS:DEN"]


def test_a_stricter_bar_is_still_available(monkeypatch):
    con = _fake_archive(monkeypatch)
    bulk.asos_stations(4000)
    assert "count(*) >= 4000" in con.sql


def test_a_partition_that_does_not_exist_yet_is_skipped(monkeypatch):
    """On 1 January the current year's file may not have been written.

    Asking for it fails the whole query rather than the one partition, which would take
    the weekly enrollment down for the hours the archive takes to catch up.
    """
    from urllib.error import HTTPError

    def fake_urlopen(req, timeout=None):
        if "year=2027" in req.full_url:
            raise HTTPError(req.full_url, 404, "not found", None, None)
        raise AssertionError("unreachable")  # pragma: no cover

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(
        bulk, "urlopen",
        lambda req, timeout=None: _Resp() if "year=2026" in req.full_url
        else fake_urlopen(req, timeout),
    )
    assert bulk._existing_year_urls(2026, 2027) == [f"{bulk.ASOS_BASE}/year=2026/data.parquet"]


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


def test_source_filter_selects_networks_and_preserves_shards():
    """Filtering must not reshuffle the split, or workers would swap stations mid-run."""
    from wxfuser.cli import filter_sources, shard_of
    from wxfuser.data.registry import load_registry

    everything = load_registry()
    live = filter_sources(everything, "ASOS,SNOTEL")

    assert 0 < len(live) < len(everything)
    assert not [s for s in live if s.id.startswith("MS:")]
    # A station keeps its worker whether or not its network was selected.
    before = {s.id: shard_of(s.id, 40) for s in everything}
    assert all(before[s.id] == shard_of(s.id, 40) for s in live)


def test_no_source_filter_is_a_no_op():
    from wxfuser.cli import filter_sources
    from wxfuser.data.registry import load_registry

    everything = load_registry()
    assert filter_sources(everything, None) is everything
    assert len(filter_sources(everything, "")) == len(everything)


# --------------------------------------------------------------- publishing the index

def test_merge_index_refuses_to_publish_an_empty_index(monkeypatch, tmp_path):
    """A failed run must not deploy an empty map over a populated one."""
    from wxfuser import cli
    from wxfuser.pipeline import core

    monkeypatch.setattr(core, "SITE_DIR", tmp_path / "site")
    monkeypatch.setattr(core, "STATE_DIR", tmp_path / "state")
    (tmp_path / "site").mkdir(parents=True)

    assert cli.cmd_merge_index(object()) == 1
    assert not (tmp_path / "site" / "index.json").exists()


def test_a_partial_run_adds_to_the_map_rather_than_replacing_it(monkeypatch, tmp_path):
    """--sources or a shard retry covers part of the registry; the rest must survive."""
    import json

    from wxfuser import cli
    from wxfuser.pipeline import core

    site, state = tmp_path / "site", tmp_path / "state"
    monkeypatch.setattr(core, "SITE_DIR", site)
    monkeypatch.setattr(core, "STATE_DIR", state)
    site.mkdir(parents=True)
    (state / "site").mkdir(parents=True)

    (state / "site" / "index.json").write_text(json.dumps(
        {"stations": [{"id": "MS:1", "status": "ok"}, {"id": "ASOS:K1", "status": "ok"}]}
    ))
    (site / "index-shard-0.json").write_text(json.dumps(
        {"stations": [{"id": "ASOS:K1", "status": "ok", "crpss_vs_raw": 0.5}]}
    ))

    assert cli.cmd_merge_index(object()) == 0
    out = {e["id"]: e for e in json.loads((site / "index.json").read_text())["stations"]}
    assert set(out) == {"MS:1", "ASOS:K1"}, "untouched station was dropped from the map"
    assert out["ASOS:K1"]["crpss_vs_raw"] == 0.5, "fresh entry did not win"
