"""Ranking stations by the population they serve, and the selection built on it.

The fleet cannot train every station that exists, so something has to decide the order.
These tests pin the two properties that ordering has to have to be worth anything: a
station beside a city outranks one in an empty valley, and a failure to fetch the
gazetteer degrades to the previous order rather than taking the run down with it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from wxfuser import cli
from wxfuser.data import prominence
from wxfuser.data.registry import Station
from wxfuser.pipeline import core

# One large city, and a smaller one 500 km away so it cannot leak across the cutoff.
CITIES = pd.DataFrame(
    {
        "name": ["Metropolis", "Smallville"],
        "lat": [40.0, 44.5],
        "lon": [-105.0, -105.0],
        "country": ["US", "US"],
        "population": [2_000_000.0, 40_000.0],
    }
)


def _station(sid, lat, lon, elev=None):
    return Station(id=sid, name=sid, lat=lat, lon=lon, elev_m=elev)


def test_a_station_in_the_city_outranks_one_in_the_hills():
    scores = prominence.served_population([40.0, 40.0], [-105.0, -103.0], CITIES)
    assert scores[0] > scores[1]


def test_population_decays_with_distance_rather_than_stopping_at_a_line():
    """Two stations either side of a city must not tie because both are 'near' it."""
    near, far = prominence.served_population([40.05, 40.4], [-105.0, -105.0], CITIES)
    assert near > far > 0


def test_nothing_within_the_cutoff_scores_zero():
    """A station in the middle of an ocean serves nobody, and should say so."""
    (score,) = prominence.served_population([0.0], [-140.0], CITIES)
    assert score == 0.0


def test_ranking_puts_the_served_stations_first_and_breaks_ties_on_elevation():
    df = pd.DataFrame(
        {
            "id": ["remote_low", "remote_high", "city"],
            "lat": [10.0, 10.0, 40.0],
            "lon": [10.0, 20.0, -105.0],
            "elev_m": [5.0, 3000.0, 1600.0],
        }
    )
    ranked = prominence.rank(df, CITIES)
    assert list(ranked["id"]) == ["city", "remote_high", "remote_low"]
    assert ranked.loc[0, "served_pop"] > 0


def test_station_sort_is_the_same_order_as_the_frame_ranking():
    stations = [
        _station("REMOTE", 10.0, 10.0, elev=100.0),
        _station("CITY", 40.0, -105.0, elev=1600.0),
    ]
    assert [s.id for s in prominence.sort_stations(stations, CITIES)] == ["CITY", "REMOTE"]


def test_a_missing_gazetteer_leaves_the_order_alone(monkeypatch, capsys):
    """Ranking is an optimisation. Losing it must not cost the run its work."""

    def boom(*_a, **_k):
        raise RuntimeError("geonames unreachable")

    monkeypatch.setattr(prominence, "sort_stations", boom)
    stations = [_station("B", 10.0, 10.0), _station("A", 40.0, -105.0)]

    assert cli.order_stations(stations, "prominence") == stations
    assert "prominence ranking unavailable" in capsys.readouterr().out


def test_registry_order_asks_for_no_gazetteer_at_all(monkeypatch):
    monkeypatch.setattr(
        prominence, "sort_stations", lambda *_a, **_k: pytest.fail("should not rank")
    )
    stations = [_station("B", 10.0, 10.0), _station("A", 40.0, -105.0)]
    assert cli.order_stations(stations, "registry") == stations


def test_untrained_skips_the_stations_that_already_have_an_archive(monkeypatch, tmp_path):
    """What lets a nightly bootstrap advance instead of restarting.

    Without it every scheduled run would begin at the same station and the tail of the
    registry would never be reached, however many nights it ran.
    """
    monkeypatch.setattr(core, "STATE_DIR", tmp_path)
    (tmp_path / "pairs").mkdir()
    (tmp_path / "pairs" / "ASOS_DEN.parquet").write_bytes(b"")

    stations = [_station("ASOS:DEN", 39.8, -104.7), _station("ASOS:BOS", 42.4, -71.0)]
    assert [s.id for s in cli.only_untrained(stations)] == ["ASOS:BOS"]


def test_the_hub_listing_names_the_stations_that_are_done(monkeypatch, tmp_path):
    """The backlog count reads filenames off the hub rather than downloading archives."""
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(core, "STATE_DIR", tmp_path)

    class FakeApi:
        def __init__(self, *_a, **_k):
            pass

        def list_repo_files(self, **_k):
            return ["catalogue/stations.parquet", "pairs/ASOS_DEN.parquet", "obs/x.parquet"]

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    assert core.trained_slugs(use_hub=True) == {"ASOS_DEN"}


def test_a_hub_listing_failure_falls_back_to_local_state(monkeypatch, tmp_path, capsys):
    """A count is worth having approximately; it is not worth failing the job for."""
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(core, "STATE_DIR", tmp_path)
    (tmp_path / "pairs").mkdir()
    (tmp_path / "pairs" / "ASOS_BOS.parquet").write_bytes(b"")

    class BrokenApi:
        def __init__(self, *_a, **_k):
            raise RuntimeError("hub unreachable")

    monkeypatch.setattr(hub, "HfApi", BrokenApi)
    assert core.trained_slugs(use_hub=True) == {"ASOS_BOS"}
    assert "could not list hub archives" in capsys.readouterr().out


# --------------------------------------------------------- the automated enrollment loop

UNIVERSE = pd.DataFrame(
    {
        "id": ["ASOS:OLD1", "ASOS:OLD2", "ASOS:CITY", "ASOS:REMOTE"],
        "name": ["old one", "old two", "in town", "far away"],
        "lat": [40.0, 40.0, 40.0, 10.0],
        "lon": [-105.0, -105.0, -105.0, 10.0],
        "elev_m": [1600.0, 1600.0, 1600.0, 100.0],
        "country": ["US", "US", "US", "XX"],
    }
)


@pytest.fixture
def bulk_enrollment(monkeypatch, tmp_path):
    """An enroll-bulk that talks to no network and writes to a scratch registry."""
    from wxfuser.data import bulk, registry

    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "stations.yaml")
    monkeypatch.setattr(bulk, "enrollable_universe", lambda **_k: UNIVERSE.copy())
    monkeypatch.setattr(prominence, "load_cities", lambda *_a, **_k: CITIES)
    registry.save_registry([_station("ASOS:OLD1", 40.0, -105.0),
                            _station("ASOS:OLD2", 40.0, -105.0)])
    return registry


def test_the_cap_counts_new_stations_rather_than_considered_ones(bulk_enrollment):
    """Two of the four are already enrolled; a cap of two must still add two.

    Capping the candidate list instead means a run whose top entries are all enrolled
    already adds nothing and exits successfully — an automated loop that stalls without
    ever reporting a problem.
    """
    assert cli.main(["enroll-bulk", "--limit", "2", "--no-bootstrap"]) == 0

    ids = [s.id for s in bulk_enrollment.load_registry()]
    assert sorted(ids) == ["ASOS:CITY", "ASOS:OLD1", "ASOS:OLD2", "ASOS:REMOTE"]


def test_enrollment_stops_when_the_backlog_is_already_full(bulk_enrollment, monkeypatch):
    """Enrolling faster than the fleet can train grows the backlog, not the site."""
    monkeypatch.setattr(cli, "_untrained_count", lambda: 900)

    assert cli.main(["enroll-bulk", "--max-backlog", "500", "--no-bootstrap"]) == 0
    assert len(bulk_enrollment.load_registry()) == 2


def test_room_left_in_the_backlog_is_what_gets_added(bulk_enrollment, monkeypatch):
    monkeypatch.setattr(cli, "_untrained_count", lambda: 499)

    assert cli.main(["enroll-bulk", "--max-backlog", "500", "--no-bootstrap"]) == 0
    ids = [s.id for s in bulk_enrollment.load_registry()]
    assert "ASOS:CITY" in ids and "ASOS:REMOTE" not in ids  # the one room allows, ranked


def test_enrollment_refuses_to_pick_an_unranked_selection(bulk_enrollment, monkeypatch):
    """Order is the whole decision here, so losing it is a reason to add nothing."""
    def boom(*_a, **_k):
        raise RuntimeError("geonames unreachable")

    monkeypatch.setattr(prominence, "rank", boom)
    with pytest.raises(SystemExit):
        cli.main(["enroll-bulk", "--limit", "2", "--no-bootstrap"])
    assert len(bulk_enrollment.load_registry()) == 2
