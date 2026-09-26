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


def test_snotel_timestamps_are_converted_from_local_standard_time_to_utc(monkeypatch):
    """AWDB stamps are station standard time. Reading them as UTC shifted every SNOTEL
    observation by 7-9 hours, pairing the afternoon maximum with the pre-dawn forecast."""
    from wxfuser.data import bulk as bulk_mod

    block = {
        "stationTriplet": "766:UT:SNTL",
        "data": [{
            "stationElement": {"elementCode": "TOBS"},
            "values": [{"date": "2026-01-11 12:00", "value": 41.0}],
        }],
    }
    out = bulk_mod._normalise_snotel(block, {"766:UT:SNTL": -8.0})
    assert out["valid_time"].iloc[0] == pd.Timestamp("2026-01-11 20:00")
    assert out["air_temp_c"].iloc[0] == pytest.approx(5.0)


def test_snotel_offset_falls_back_to_pst_when_metadata_is_unreachable(monkeypatch):
    from wxfuser.data import obs as obs_mod

    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(obs_mod, "_http", boom)
    monkeypatch.setattr(obs_mod, "_snotel_offsets", {})
    assert obs_mod.snotel_utc_offsets(["1:CO:SNTL"]) == {"1:CO:SNTL": -8.0}


def test_rebuild_discards_history_and_demands_a_bootstrap(tmp_path, monkeypatch):
    from wxfuser import cli
    from wxfuser.data.registry import Station
    from wxfuser.pipeline import core

    monkeypatch.setattr(core, "STATE_DIR", tmp_path)
    st = Station(id="766:UT:SNTL", name="Snowbird", lat=40.57, lon=-111.66)
    for p in (core.pairs_path(st), core.obs_path(st)):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"stale")
    core.discard_history(st)
    assert not core.pairs_path(st).exists() and not core.obs_path(st).exists()
    core.discard_history(st)  # nothing left to delete is not an error

    with pytest.raises(SystemExit, match="needs --bootstrap"):
        cli.main(["refresh", "--rebuild"])


def test_repair_keeps_forecasts_and_rejoins_corrected_observations(tmp_path, monkeypatch):
    from wxfuser.data import pairs as pairs_mod
    from wxfuser.data.registry import Station
    from wxfuser.pipeline import bulk_run, core

    monkeypatch.setattr(core, "STATE_DIR", tmp_path)
    st = Station(id="766:UT:SNTL", name="Snowbird", lat=40.57, lon=-111.66,
                 variables=["air_temp_c"])
    hours = pd.date_range("2026-01-11", periods=24, freq="h")
    fc = pd.DataFrame({"model": "gfs_seamless", "valid_time": hours, "lead_h": 24,
                       "lead_source": "prev_runs", "fc_air_temp_c": np.arange(24.0)})
    shifted = pd.DataFrame({"station_id": st.id, "valid_time": hours - pd.Timedelta(hours=8),
                            "air_temp_c": np.arange(24.0), "source": "SNOTEL"})
    pairs_mod.write_archive(pairs_mod.build_pairs(fc, shifted, st.id, ["air_temp_c"]),
                            core.pairs_path(st))
    corrected = shifted.assign(valid_time=hours)
    for c in OBS_COLUMNS:
        if c not in corrected:
            corrected[c] = np.nan
    monkeypatch.setattr(bulk_run, "gather_observations_bulk",
                        lambda stations, start, end: {st.id: corrected[OBS_COLUMNS]})

    stored = len(pairs_mod.read_archive(core.pairs_path(st)))
    assert stored == 16  # the 8-hour shift left only the overlapping hours paired

    report = bulk_run.repair_observations([st])
    assert report[0]["status"] == "repaired"
    out = pairs_mod.read_archive(core.pairs_path(st))
    # Every stored forecast row survives, and forecast and observation now agree hour
    # for hour. Rows the shift never stored cannot be recovered, and are not invented.
    assert len(out) == stored
    assert np.allclose(out["fc_air_temp_c"], out["obs_air_temp_c"])


def test_repair_leaves_a_station_alone_when_observations_do_not_arrive(tmp_path, monkeypatch):
    from wxfuser.data import pairs as pairs_mod
    from wxfuser.data.registry import Station
    from wxfuser.pipeline import bulk_run, core

    monkeypatch.setattr(core, "STATE_DIR", tmp_path)
    st = Station(id="1:CO:SNTL", name="x", lat=40.0, lon=-106.0)
    archive = pd.DataFrame({"station_id": st.id, "model": "m",
                            "valid_time": pd.date_range("2026-01-01", periods=3, freq="h"),
                            "lead_h": 3, "lead_source": "hist", "fc_air_temp_c": 1.0,
                            "obs_air_temp_c": 2.0, "obs_source": "SNOTEL"})
    pairs_mod.write_archive(archive, core.pairs_path(st))
    monkeypatch.setattr(bulk_run, "gather_observations_bulk", lambda *a: {})
    assert bulk_run.repair_observations([st])[0]["status"] == "untouched"
    assert len(pairs_mod.read_archive(core.pairs_path(st))) == 3


def test_gauge_jitter_is_not_precipitation():
    """12.3, 12.4, 12.3 in is no rain. Clipped differences called it 2.5 mm, which at
    Snowbird booked 2,073 mm over a summer whose gauge gained 183 mm."""
    from wxfuser.data.obs import gauge_increments

    jitter = pd.Series([312.4, 314.9, 312.4, 314.9, 312.4, 312.4])
    assert gauge_increments(jitter).fillna(0).sum() == pytest.approx(0.0)

    rain = pd.Series([100.0, 100.0, 102.5, 105.0, 104.9, 105.0])
    inc = gauge_increments(rain)
    assert np.isnan(inc.iloc[0])
    assert inc.sum() == pytest.approx(5.0, abs=0.11)


def test_gauge_reset_starts_a_new_segment():
    from wxfuser.data.obs import gauge_increments

    wy = pd.Series([900.0, 902.5, 0.0, 0.0, 2.5])  # 1 October zeroes the gauge
    inc = gauge_increments(wy)
    assert inc.iloc[1] == pytest.approx(2.5)
    assert np.isnan(inc.iloc[2])  # no increment across a reset
    assert inc.iloc[4] == pytest.approx(2.5)


def test_snotel_requests_stay_under_awdbs_size_limit_and_stitch(monkeypatch):
    """AWDB answers 400 past ~3,600 station-days, which silently failed every deep
    SNOTEL backfill. Requests are split in time, and the pieces are stitched before
    precipitation is differenced, so no increment is lost at a window edge."""
    from datetime import date, timedelta
    from urllib.parse import parse_qs, urlparse

    from wxfuser.data import bulk as bulk_mod

    seen = []

    def fake(url, timeout=120):
        q = parse_qs(urlparse(url).query)
        stations = q["stationTriplets"][0].split(",")
        b, e = (date.fromisoformat(q[k][0]) for k in ("beginDate", "endDate"))
        days = (e - b).days + 1
        seen.append(len(stations) * days)
        assert len(stations) * days <= 3600, "AWDB would refuse this"
        hours = pd.date_range(b, e + timedelta(days=1), freq="h", inclusive="left")
        base = pd.Timestamp("2025-01-01")
        return [{"stationTriplet": st, "data": [{
            "stationElement": {"elementCode": "PREC"},
            # 0.01 in per hour, cumulative from a fixed origin
            "values": [{"date": h.strftime("%Y-%m-%d %H:%M"),
                        "value": round(0.01 * (h - base) / pd.Timedelta(hours=1), 2)}
                       for h in hours]}]} for st in stations]

    monkeypatch.setattr(bulk_mod, "_http_json", fake)
    monkeypatch.setattr(bulk_mod, "snotel_utc_offsets", lambda t: {x: 0.0 for x in t})
    triplets = [f"{i}:UT:SNTL" for i in range(40)]
    out = bulk_mod.snotel_observations(triplets, date(2025, 1, 1), date(2025, 12, 31))
    assert len(seen) > 1
    one = out[out["station_id"] == "0:UT:SNTL"].sort_values("valid_time")
    assert len(one) == 365 * 24
    # Only the very first hour lacks an increment; window edges do not.
    assert one["precip_1h_mm"].isna().sum() == 1
    assert one["precip_1h_mm"].dropna().round(3).eq(0.254).all()
